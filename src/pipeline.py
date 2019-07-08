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

import tensorflow as tf
import numpy as np
import os
import cv2

import random

import data_tools
import targets
import tiling

from shapely.geometry.polygon import Polygon


def _generate_tiles(tile_size,image_files,gt_files):
    """
    Creates a tile generator that takes a map from memory, crops
    a tile of size tile_size, and finds all ground truths inside that tile

    Parameters
      tile_size: the scalar length of the square tile side
    Returns 
      generator: a generator that yields a cropped image tile and an 
                   offset-adjusted list of ground truth rectangles, as returned 
                   by tiling.get_random_tile().
    """

    assert len(image_files) == len(gt_files)

    # load the entire training data set into memory. This is
    # memory-intensive, but we want to do it so we're not hitting the
    # file system for every tile, which is a huge net drag on training
    # throughput when the source images are quite large.
    
    images = [ cv2.imread(fname)[:,:,::-1] for fname in image_files ] # BGR->RGB

    groundtruths = [ data_tools.parse_boxes_from_json(fname)
                     for fname in gt_files ]
    groundtruth_points = [ gt[0] for gt in groundtruths]
    groundtruth_polys  = [ gt[1] for gt in groundtruths]
    groundtruth_labels = [ gt[2] for gt in groundtruths] # NB: Currently unused

    while True:
        try:
            # choose one random source image
            random_index = random.randint(0, len(images)-1)
            image = images[random_index]
            gt_points = groundtruth_points[random_index]
            gt_polys = groundtruth_polys[random_index]
            tile, mod_ground_truth = tiling.get_random_tile( image,
                                                             gt_points,
                                                             gt_polys,
                                                             tile_size)
            yield tile, mod_ground_truth
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            # continue


def get_targets(tile, ground_truths, tile_shape):
    """
    Given a tile and ground truths, generate the geometry and score maps
    and the training mask using targets.generate()

    Parameters:
      tile         : cropped map image
      ground_truths: list of ground truth rectangle coordinates intersecting 
                      the tile
      tile_shape   : a two-element tuple [tile_height,tile_width]
    Returns
      tile       : given input
      score_map  : tile_shape/4 x 1 array of ground truth labels
      geo_map    : tile_shape/4 x 5 array of target geometries
      train_mask : tile_shape/4 x 1 array of loss-inclusion weights
    """

    score_map, geo_map, train_mask = targets.generate(tile_shape, ground_truths)
    
    return tile, score_map, geo_map, train_mask



def _get_filenames(image_path, gt_path, file_patterns, image_ext, gt_ext):
    """
    Construct pairs of complete file names from the base path and patterns for
    images and their corresponding ground truths

    Parameters:
      image_path    : Directory containing the images
      gt_path       : Directory containing the ground truth files
      file_patterns : List of wildcard basename patterns for reading input files
      image_ext     : Extension of image files (appended to file_patterns)
      gt_ext        : Extension of ground truth files
    Returns: 
      img_files   : A list of image file names (with path and extension)
      gt_files      : A list of ground truth file names (with path and extension)
    """
    img_files = data_tools.get_filenames( image_path, file_patterns, image_ext)
    gt_files  = data_tools.get_paired_filenames(img_files,gt_path,gt_ext)

    return img_files, gt_files

    
def get_dataset(image_path, gt_path, file_patterns,
                image_ext='tiff',
                gt_ext='json',
                tile_size=512,
                batch_size=16 ):
    """
    Returns a batched dataset

    Parameters:
      image_path    : Directory containing the images
      gt_path       : Directory containing the ground truth files
      file_patterns : List of wildcard patterns for reading input files
      image_ext     : Extension of image files (to append to file_patterns)
      gt_ext        : Extension of label files (to append to file_patterns)
      tile_size     : Side length of the square tile
      batch_size    : Number of tiles per batch
    Returns: 
      ds : a tf.data.Dataset object that can generate tile batches for training
    """

    prefetch_buffer_size = 1 # num of batches to prefetch

    # Expand wildcards, prepend paths, and append extensions
    # to get paired input files
    image_files,gt_files = _get_filenames( image_path, gt_path, file_patterns,
                                           image_ext, gt_ext)
    
    # Begin with a generator that produces an image tile and the intersecting
    # ground truth rectangles, translated to the tile's coordinates
    # (rather than the whole image)
    
    # need to provide (not invoke) a function that returns a generator
    ds = tf.data.Dataset.from_generator (
        generator=lambda : _generate_tiles(tile_size,image_files,gt_files),
        output_types=(tf.float32, tf.float32), 
        output_shapes=(tf.TensorShape([tile_size, tile_size, 3]), 
                       tf.TensorShape([4, 2, None])))
    
    # Calculate the output targets for the ground truths
    ds = ds.apply(
        tf.data.experimental.map_and_batch (
            lambda tile, ground_truth: tf.py_func (
                func = get_targets, 
                inp  = [ tile, ground_truth, (tile_size, tile_size) ],
                #       tile        score_map   geo_map     train_mask
                Tout = [tf.float32, tf.float32, tf.float32, tf.float32] ), 
            batch_size )
    )

    # Pack the results to feat_dict,labels for the Estimator, 
    # explicitly giving tile shape for downstream model (keras resnet),
    # otherwise the shape is unknown
    ds = ds.map(
        lambda tile, score_map, geo_map, train_mask:
        ({'tile': tf.reshape(tile, [batch_size, tile_size, tile_size, 3]),
          'score_map': score_map,
          'geo_map': geo_map,
          'train_mask': train_mask}, geo_map))

    ds = ds.prefetch(prefetch_buffer_size)
    
    return ds

def get_prediction_dataset(image_file, rects_file,**kwargs):
    """
    Get a dataset from the lines in rects_file where each element is a
    tuple with the predicted rectangle (4x2 float32 tensor with points in CCW
    and bottom-left as the first point) and the cropped, rotated image (RGB uint8)
    corresponding to that rectangle as extracted from image_file.

    Parameters
      image_file: Path to the image to read rectangles from
      rects_file: Path to the file containing predicted rectangles
      **kwargs:   Optional arguments to data_tools.normalize_box
    Returns
      dataset:    tf.data.Dataset with (rectangle, image) tuples 
    """
    
    def gen_rect_image(): # Generator for yielding (rect,cropped) tuples
        image = cv2.imread(image_file)[:,:,::-1] # BGR->RGB
        rects = data_tools.parse_boxes_from_text(rects_file,slice_first=True)[0]

        for rect in rects:
            cropped = data_tools.normalize_box(image,rect,**kwargs)
            yield rect, cropped

    # Construct the Dataset object from the generator
    dataset = tf.data.Dataset.from_generator (
        generator=gen_rect_image,
        output_types=(tf.float32, tf.uint8),
        output_shapes=(tf.TensorShape([4,2]),
                       tf.TensorShape([None,None,3])))

    return dataset
