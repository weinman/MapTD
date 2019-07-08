# MapTD
# Copyright (C) 2018 Nathan Gifford, Jerod Weinman, Abyaya Lamsal
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

# visualize.py -- Utilities for drawing boxes on a given image and saving the
#   result as a raster. Displays result as a command-line program, e.g.,
#     python visualize.py image_file box_file

import numpy as np
import os
import sys
import matplotlib.pyplot as plt


baseline_color='blue'
line_color='red'
box_linewidth=0.2

def load_boxes(filename):
    """
    Retrieves boxes from given text file, expected to have at least 8 
    comma-separated numbers per line

    Parameters
       filename: name of the text file that contains the boxes
    Returns
       polygons: a list of boxes, where a box is a list of four corners and a
                   corner is a list of two numbers representing the column and 
                   row (in that order) of the point in the image
    """
    polygons = list()
    with open(filename, 'r') as fd:
        for line in fd.readlines():
            v = line[:-2].split(',')
            p1 = [float(v[0]), float(v[1])]
            p2 = [float(v[2]), float(v[3])]
            p3 = [float(v[4]), float(v[5])]
            p4 = [float(v[6]), float(v[7])]
            verts = [p1, p2, p3, p4]
            polygons.append(verts)

    return polygons


def draw_line(p0,p1,clr):
    """Draw a line on the current PyPlot axes between two points

    Parameters
      p0  : list as (x,y) pair for one line endpoint
      p1  : list as (x,y) pair for one line endpoint
      clr : valid color option to pyplot.plot (a string)
    """
    plt.plot([p0[0],p1[0]],[p0[1],p1[1]],
             linewidth=box_linewidth, color=clr,
             scalex=False,scaley=False)


def save_image(image, boxes, output_base):
    """
    Display outline of each each box on top of image (with the first segment
    in baseline_color and the others in line_color) writing result to file
    "output_base.png"
    
    Parameters
       image       : image to render boxes upon (numpy array)
       boxes       : list of boxes that correspond to the image (a box is a list
                       of four corners and a corner is a list of two numbers 
                       representing the column and  row (in that order) of the 
                       point in the image
       output_base : complete path and basename (sans extension) of 
                       rendered image to write
    """
    render_boxes(image, boxes)
    save_as = output_base+'.png'
    print 'Saving image as', save_as
    plt.savefig(save_as, dpi=1000, bbox_inches='tight')


def render_boxes(image, boxes):
    """
    Render outline of each each box on top of image (with the first segment
    in baseline_color and the others in line_color) writing result to file
    "output_base.png"
    
    Parameters
       image       : image to render boxes upon (numpy array)
       boxes       : list of boxes that correspond to the image (a box is a list
                       of four corners and a corner is a list of two numbers 
                       representing the column and  row (in that order) of the 
                       point in the image
    
    Parameters:
       im: the image (array)
       boxes: the boxes that correspond to the image
              a box is a list of four corner vertex pairs
    """
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(image)
    for n in range(len(boxes)):
        box = boxes[n]
        draw_line( box[0],box[1], baseline_color)
        draw_line( box[1],box[2], line_color)
        draw_line( box[2],box[3], line_color)
        draw_line( box[3],box[0], line_color)



def main(image_path, ground_truth_file):
    """Load and display boxes on an image

    Parameters
      image_path: path to the image on which to display the boxes
      truth     : name of the text file that contains the boxes to draw
    """
    boxes = load_boxes(ground_truth_file)
    image = plt.imread(image_path)
    render_boxes(image, boxes)
    plt.show()

if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2])
