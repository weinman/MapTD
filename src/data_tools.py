# MapTD
# Copyright (C) 2018 Nathan Gifford
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# data_tools.py -- Reads ground truth data, converts polygons to rectangles, and

import glob
import os
import json
import numpy as np
import cv2
from shapely.geometry.polygon import Polygon



def parse_boxes_from_json(filename,slice_first=False):
    """
    Reads a ground truth file and returns a list, where the first element is
    the list of ground truths in the form of vertices and the second element
    is the list of ground truths in the form of Shapely Polygon objects

    Parameters
      filename: the name of the file to parse
      slice_first : Whether to shape data as Nx4x2 when True or 
                      4x2xN when False (default 
    Returns
      points   : 4x2xN array of boxes in the JSON file
      polygons : List of corresponding Shapely polygons 
      labels   : List of ground truth label strings
    """
    points = list()
    labels = list()
    
    with open(filename) as fd:
        data = json.load(fd)

    for entry in data:
        for item in entry['items']:

            if item['text']:
                text = item['text'].encode('utf-8')
            else:
                text = None
                
            labels.append(text)
            verts = item['points']

            verts = np.asarray(verts, dtype=np.float32)
            verts,_ = convert_polygon_to_rectangle(verts) # Assumes CCW points 

            points.append(verts)

    # Sort by area (decreasing) as a proxy for height so that output
    # geometries of smaller rectangles do not get overwritten by
    # larger rectangles during target generation
    points_labels = sorted(zip(points,labels),
                           key=lambda verts_label: -Polygon(verts_label[0]).area)
    points,labels = zip(*points_labels)

    # Construct paired polygons for later intersection calculation
    polygons = [Polygon(verts).convex_hull for verts in points]
    
    points= np.asarray(points)

    if not slice_first:
        points = np.transpose(points, (1, 2, 0)) # Turn Nx4x2 to 4x2xN

    return [points, polygons, labels]



def parse_boxes_from_text(filename,slice_first=False):
    """Reads a file where each line contains the eight points of a
    rectangle in x0,y0,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7
    format and returns a list, where the first element is the list of
    ground truths in the form of vertices and the second element is
    the list of ground truths in the form of Shapely Polygon objects

    Parameters
      file        : Path to the file to parse
      slice_first : Whether to shape data as Nx4x2 when True or 
                      4x2xN when False (default 
    Returns
      points   : Array of box vertices from the file; shape is 4x2xN if 
                   slice_first=False [default], otherwise Nx4x2
      polygons : List of corresponding Shapely polygons
      labels   : List of corresponding labels from the text file lines
      scores   : List of corresponding prediction scores from text file lines
    """
    points = list()
    polygons = list()
    labels = list()
    scores = list()
    
    with open(filename, 'r') as fd:
        for line in fd:
            v = line[:-1].split(',')
            p1 = [float(v[0]), float(v[1])]
            p2 = [float(v[2]), float(v[3])]
            p3 = [float(v[4]), float(v[5])]
            p4 = [float(v[6]), float(v[7])]
            verts = np.asarray([p1, p2, p3, p4],dtype=np.float32)
            label = json.loads(str(v[8]))
            score = float(v[9]) if len(v)>9 else 0
            
            points.append(verts)
            poly = Polygon(verts)
            polygons.append(poly)
            labels.append(label)
            scores.append(score)
    points = np.asarray(points)
    scores = np.asarray(scores)
    if not slice_first:
        points = np.transpose(points, (1, 2, 0)) # Turn Nx4x2 to 4x2xN

    return [points, polygons, labels, scores]


def get_files_from_dir(dir_path):
    """
    Gets a list of file names from a directory

    Parameters
      dir_path : Path of a file or the directory to read
    Returns
      files : Sorted list of paths to the given file or files in given directory
    """
    files = []
    if os.path.isdir(dir_path):
        names = os.listdir(dir_path)
        files = [os.path.join(dir_path, name) for name in names]
    elif os.path.isfile(dir_path):
        files = [dir_path]
    else:
        raise ValueError('invalid input directory or file')
    
    return sorted(files)

def get_filenames(path,file_patterns,ext):
    """
    Construct complete file names from the base path and patterns for
    images

    Parameters
      path          : Directory containing the files
      file_patterns : List of wildcard basename patterns for reading input files
      ext           : Extension of files (appended to file_patterns)
    Returns 
      filenames : A list of file names (with path and extension)
    """

    # List of lists ...
    filenames = [ glob.glob( os.path.join( path,
                                           file_pattern+'.'+ext ) )
                   for file_pattern in file_patterns]
    # flatten
    filenames = [ filename for sublist in filenames
                    for filename in sublist]
    
    return filenames

def get_paired_filenames(filenames,paired_path,ext):
    """
    Construct the paired complete file names from a list of filenames
    with a new base path and extension

    Parameters
      filenames   : List of filenames to extract basenames (sans extension) from
      paired_path : Directory containing the files to be paired
      ext         : Extension of files to append
    Returns
      paired_filenames : A list of file names (with path and extension)
    """

    # Get the corresponding files with matching basename in paired ordering
    base_names = [ os.path.splitext(os.path.basename(fname))[0]
                   for fname in filenames ] # Strip trailing extension
    paired_filenames = [ os.path.join(paired_path, base_name+'.'+ext)
                         for base_name in base_names ]
    return paired_filenames


def set_correct_order(points, target_point, from_quad=True):
    """

    Find the correct bottom left point of the box generated by opencv and make it
    the first element of the list. If the original polygon from which the points
    are derived was a quadrilateral, the closest of all four points are 
    considered. Otherwise, the closest of the first (in a CCW ordering) points
    along the longest edge are considered.
    
    Parameters
      points       : 4x2 array of the points of the box generated by OpenCV
      target_point : Bottom-left (in text semantics) point from ground truth
      from_quad    : Whether the original polygon for the box was a quadrilateral
    Returns
      points : A barrel shift of input points so bottom-left is first
    """
    if from_quad:
        # Distances of every vertex from the target
        distances = np.linalg.norm( points - target_point, axis=1)
        closest = np.argmin(distances)
    else:
        # find the left points of the long sides
        dist01 = np.linalg.norm(points[0] - points[1])
        dist12 = np.linalg.norm(points[1] - points[2])
        
        if (dist01 > dist12):
            candidates = [0,2] # First points along two longest edges
        else:
            candidates = [1,3]
        # find closer candidate
        dist0 = np.linalg.norm(points[candidates[0]] - target_point)
        dist1 = np.linalg.norm(points[candidates[1]] - target_point)
        
        closest = candidates[0] if dist0 < dist1 else candidates[1]

    # shift so closest point is first
    points = np.concatenate((points[closest:],points[:closest]),axis=0)
    return points


def convert_polygon_to_rectangle(poly):
    """
    Converts an arbitrary polygon to the rectangle of best fit. The rectangle's
    four points are counter-clockwise ordered to with the first point being 
    closest to the polygon's first point.

    Parameters
      poly : Counter-clockwise Nx2 array of points of bounding polygon, the first 
               point being the bottom-left in text semantics
    Returns
      points : Counter-clockwise 4x2 array of points of minimum bounding 
                 rectangle, the first point being the bottom-left in text
                 semantics
      box    : Corresponding box tuple ((x,y),(w,h),ang) from OpenCV minAreaRect
    """
    bottom_left = poly[0]
    box = cv2.minAreaRect(poly)
    points = cv2.boxPoints(box) # OpenCV gives points in clockwise order
    
    # Convert clockwise to the counter-clockwise expected by targets.generate()
    points = np.flip(points,axis=0)
    points = set_correct_order(points, bottom_left, poly.shape[0]==4)
    
    return points,box

def normalize_box(image, rect, max_height_width_ratio=None):
    """ 
    Generate a cropped, axis-aligned version of the given rectangular box from 
    the given image. The vertices of the rectangle must given in 
    counter-clockwise order starting from the bottom-left (in text semantics) as
    given by set_correct_order and (by extension) convert_polygon_to_rectangle.

    Parameters
      image : Input image to extract the normalized box from
      rect  : 4x2 array of rectangle vertices
      max_height_width_ratio: If height/width of box exceeds this value, the image
                                box is extended evenly on the left and right to
                                meet the threshold. [Default: None]
    Returns
      crop : Cropped and rotated image
    """
    center = tuple(np.mean(rect,axis=0))
    vech   = rect[1] - rect[0]
    vecv   = rect[2] - rect[1]
    width  = np.linalg.norm(vech)
    height = np.linalg.norm(vecv)
    angle  = np.rad2deg( np.arctan2(vech[1],vech[0]) )

    # Adjust for minimum width (if necessary)
    if max_height_width_ratio and height/width > max_height_width_ratio:
        width = np.ceil(height / max_height_width_ratio)

    # Cropped rotation adapted from https://stackoverflow.com/a/11627903 
    # by rroowwllaanndd and Suuuehgi.
    # Used under Creative Commons Attribution-Share Alike (3.0) license.
    x0 = int(center[0] - width/2)  # Cropping boundaries in rotated image
    y0 = int(center[1] - height/2)

    x1 = x0 + int(round(width))
    y1 = y0 + int(round(height))

    if x1<=0 or y1<=0 or min( [ y1-y0-1, x1-x0-1] ) < 1:
        # check whether rectangle outside the image bounds or too small    
        # rather than fail, hope someone else detects and return a dummy  image
        return np.zeros([1,1,3],dtype=np.uint8)
    if x0<0: # If the left side goes off the edge,
        # bump the rectangle rightward (to preserve min width) or else just trim
        x1 = (x1 - x0) if max_height_width_ratio else x1
        x0 = 0
        
    matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    image_rot = cv2.warpAffine(src=image, M=matrix, dsize=(x1,y1))
    image_rot_crop = image_rot[y0:y1, x0:x1]

    return image_rot_crop
