# MapTD
# Copyright (C) 2018 Nathan Gifford, Abyaya Lamsal, Jerod Weinman
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

import os
import random
import importlib

import numpy as np
from shapely.geometry.polygon import Polygon


def crop_ground_truths(x, y, tile_size, gt_coords, gt_polys):
    """
    Takes a tile location+dimensions and finds all ground truths that
    intersect that tile

    param x: the x-coord of the top-left vertex of the tile
    param y: the y-coord of the top-left vertex of the tile
    param tile_size: the length of the tile side
    param gt_coords: a list of the ground truths in the form of the vertices
    param gt_polys: a list of the ground truths in the form of Shapely polygons
    returns: a list of shifted ground truth coordinates that intersect the tile
    """
    tile = Polygon( [ (x, y),
                      (x, y+tile_size),
                      (x+tile_size, y+tile_size),
                      (x+tile_size, y) ] )
    
    intersects_tile = [tile.intersects(gt) for gt in gt_polys]
    intersects_tile = np.asarray( intersects_tile )
        
    keep_indices = np.argwhere( intersects_tile )

    # Hack because squeezing a 1x1 matrix will return a scalar,
    # which will break the code
    if (len(keep_indices) == 1):
        keep_indices = keep_indices[0]
    else:
        keep_indices = np.squeeze(keep_indices)

    keep_gt = np.asarray( gt_coords[:,:,keep_indices] )
    keep_gt[:,0,:] = keep_gt[:,0,:] - np.asarray([x])
    keep_gt[:,1,:] = keep_gt[:,1,:] - np.asarray([y])
    # TODO: Sort ground truths by area for marking shrunk away don't
    # care regions correctly when generating targets.

    return keep_gt


def get_random_tile(img, gt_coords, gt_polys, tile_size):
    """
    Takes an image, crops out a tile, and calculates the corresponding ground 
    truths by finding the intersecting rectangles

    Parameters
      img      : the image to take a tile from
      gt_coords: the list of ground truths in the form of vertices
      gt_polys : the list of ground truths in the form of Shapely polygons
      tile_size: the length of the tile side
    Returns 
      tile         : cropped image region
      ground_truth : offfset-adjusted list of ground truth polygons
    """
    h = len(img)
    w = len(img[0])
    x = random.randint( 0, w-tile_size )
    y = random.randint( 0, h-tile_size )
    tile = img[y:y+tile_size,x:x+tile_size]
    ground_truth = crop_ground_truths( x, y, tile_size, gt_coords, gt_polys )

    return tile, ground_truth
