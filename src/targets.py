# MapTD
# Copyright (C) 2018 Jerod Weinman, Nathan Gifford, Abyaya Lamsal, 
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

import numpy as np
from numpy.linalg import norm
import cv2

"""In all the routines below, we assume the rectangle coordinates are
sorted in counter-clockwise order, so that the first point corresponds
to the left point of the text baseline (presuming a left-to-right reading
order) as in these examples.


  3 *---------* 2         3 *                 2 *
    | T E X T |            / `                 / `* 1
  0 *---------* 1       0 * T `               / r/
                           ` h `             / a/
                            ` e * 2       3 * B/
                             ` /             `* 0
                              * 1           
"""
def get_angle ( line_segment ):
    """Angle of the line segment and an array of its cosine and sin
    
    Parameters:
       line_segment: A 1x2 np array of the directed line segment (x,y)
    Returns:
       angle: The scalar angle of the directed line segment (in radians)
       cos_sin: A 1x2 np array containing the cosine and sine of that angle
"""
    angle =  np.arctan2( line_segment[1], line_segment[0] )
    cos_sin = np.array([np.cos(angle), np.sin(angle)])
    return angle, cos_sin


def shrink_rect( rect, shrink_ratio=0.3 ):
    """ Shrink the edges of a rectangle by a fixed relative factor. The
        effect should be equivalent to scaling the height and width of a
        rotated box represented as a center, size, and rotated angle.
    
    Parameters:
        rect: A 4x2 numpy array indicating the coordinates of the four rectangle
              vertices
        shrink_ratio: A scalar in (0,0.5) indicating by how much to move each 
                      point along the line segment representing a rectangle 
                      side. [default 0.3]
    Returns:
        shrunk: A 4x2 numpy array with the modified rectangle points
    """

    # Modeled on Eq. (3) in Zhou et al. (EAST), but the mod is outside the +/- 1
    # due to Python's zero-based indexing
    reference_lengths = [ min( norm( rect[c] - rect[(c+1)%4] ),
                               norm( rect[c] - rect[(c-1)%4] ) ) 
                          for c in range(4) ]

    shrunk = rect.copy().astype(np.float32) # Create a clean copy for mutation

    # Find the longer pair of edges --- 
    # {<p0,p1>,<p3,p2>} versus {<p0,p3>,<p1,p2>}
    len_01_32 = norm(rect[0] - rect[1]) + norm(rect[3] - rect[2])
    len_03_12 = norm(rect[0] - rect[3]) + norm(rect[1] - rect[2])

    # Local helper function to shrink a line segment <start,end>
    def shrink(start,end):
        cos_sin = get_angle(rect[end]-rect[start])[1]
        shrunk[start] += shrink_ratio * reference_lengths[start] * cos_sin
        shrunk[end]   -= shrink_ratio * reference_lengths[end]   * cos_sin
    # Local helper function to shrink all edges in given order
    def shrink_edges(edges):
        for edge in edges:
            shrink(edge[0],edge[1])

    # Move the longer axes first then shorter axes 
    if len_01_32 > len_03_12:
        shrink_edges( [[0,1],[3,2],[0,3],[1,2]] )
    else:
        shrink_edges( [[0,3],[1,2],[0,1],[3,2]] )

    return shrunk


def dist_to_line(p0, p1, points):
    """ Calculate the distance of points to the line segment <p0,p1> """
    norm1 = norm( p1-p0 )
    if norm1 == 0:
        print p0, p1
        norm1 = 1.0
    return np.abs( np.cross(p1-p0, points-p0) / norm1 )


def generate(image_size, rects):
    """ Generate the label maps for training from the preprocessed rectangles 
        intersecting the cropped subimage. 

    Parameters:
       image_size: A two-element tuple [image_height,image_width]
       rects: An 4x2xN numpy array containing the coordinates of the four 
              rectangle vertices. The zeroth dimension runs clockwise around the
              rectangle (as given by sort_rectangle), the first dimension is 
              (x,y), and the last dimension is the particular rectangle.
    Returns:
       score_map : An image_size/4 array of ground truth labels (in {0,1}) for 
                     shrunk versions of the given rectangles
       geo_map   : An image_size/4 x 5 array of geometries for the shrunk 
                     rectangles; the final dimension contains the distances to 
                     the top, left, bottom, and right rectangle eges, as well as
                     the oriented angle of the top edge in [0,2*pi)
       train_mask: An image_size/4 x 1 array of weights (in {0,1}) to include in
                     the loss function calculations.
"""

    # ---------------------------------------------------------------------------
    # Set up return values 
    
    # Where a given rectangle is located
    rect_mask = np.zeros( image_size, dtype=np.uint8) 

    # Pixel-wise positive/negative class indicators for loss calculation
    score_map  = np.zeros( image_size, dtype=np.uint8 ) 
    
    # Distances to four rectangle edges and angle
    geo_map = np.zeros( [image_size[0],image_size[1],5], dtype=np.float32)

    # Which pixels are used or ignored during training. Initially 2 (unknown)
    training_mask = 2 * np.ones( image_size, dtype=np.uint8 )

    #---------------------------------------------------------------------------
    # Iterate over rectangles:

    for r in xrange(rects.shape[2]):
        rect = rects[:,:,r]
        # Shrink the rectangle, and put in a fillPoly-friendly format
        shrunk_rect = shrink_rect( rect ).astype(np.int32)[np.newaxis,:,:]

        
        # Set ground truth pixels to detect
        cv2.fillPoly(score_map, shrunk_rect, 1) 

        # Invariant: rect_mask all 0 before this
        cv2.fillPoly(rect_mask, rect.astype(np.int32)[np.newaxis,:,:], 2)
        cv2.fillPoly(rect_mask, shrunk_rect, 1)

        # If we wanted to ignore rectangles that were too small, 
        # we might do so here    
        #rect_h = min( norm( rect[0]-rect[3]), norm(rect[1]-rect[2]))
        #rect_w = min( norm( rect[0]-rect[1]), norm(rect[2]-poly[3]))
        #if min(rect_h, rect_w) < MIN_POLY_SIZE:
        #    cv2.fillPoly(training_mask, 
        #                 rect.astype(np.int32)[np.newaxis, :, :], 0)

        yx_in_shrunk_rect = np.argwhere( rect_mask == 1 )
        xy_in_shrunk_rect = yx_in_shrunk_rect[:,::-1]
        rows = yx_in_shrunk_rect[:,0]
        cols = yx_in_shrunk_rect[:,1]

        # Unlike argman/EAST, ignore the pixels in the ground truth rectangle
        # that were shrunk away, rather than treating them as "negative" labels
        clear_eroded = rect_mask==2 # eroded area

        # Also want to clear any new positive labels that are already labeled,
        # to avoid learning confusion about potentially conflicting outputs
        clear_intersection = np.logical_and( rect_mask!=0, 
                                             training_mask!=2 )

        # Set to train on shrunk_rect
        cv2.fillPoly(training_mask, shrunk_rect, 1) 


        to_clear = np.logical_or( clear_eroded, clear_intersection)
        training_mask[np.nonzero(to_clear)] = 0

        # top, left, bottom, right, angle
        geo_map[rows,cols,0]=dist_to_line( rect[2], rect[3], xy_in_shrunk_rect)
        geo_map[rows,cols,1]=dist_to_line( rect[0], rect[3], xy_in_shrunk_rect)
        geo_map[rows,cols,2]=dist_to_line( rect[0], rect[1], xy_in_shrunk_rect)
        geo_map[rows,cols,3]=dist_to_line( rect[1], rect[2], xy_in_shrunk_rect)
        geo_map[rows,cols,4]=get_angle( rect[1] - rect[0])[0]

        rect_mask[:,:] = 0 # Restore invariant

    # Set to train on any locations not modified
    training_mask[np.nonzero(training_mask==2)] = 1
    
    # The loss function will want the score/mask to be floats.
    # We store as uint8 to conserve space, converting only after downsampling.
    
    return  score_map[::4,::4, np.newaxis].astype(np.float32), \
        geo_map[::4,::4,:], \
        training_mask[::4,::4, np.newaxis].astype(np.float32)
