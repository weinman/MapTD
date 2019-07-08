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

# predict.py -- Run a trained model on images to produce bounding box
#   predictions and (optionally) image visualizations

import cv2
import math
import os
import numpy as np
import tensorflow as tf

import lanms
import model
import data_tools
import visualize


tf.app.flags.DEFINE_string('model', '../data/model',
                           """Directory for model checkpoints""")

tf.app.flags.DEFINE_integer('tile_size',4096,
                            """Tile size for image processing""")
tf.app.flags.DEFINE_integer('tile_overlap',1024,
                            """Tile overlap for image processing""")

tf.app.flags.DEFINE_string('image_path','../data/images',
                           """Base directory for image training data""")
tf.app.flags.DEFINE_string('filename_pattern','*',
                           """File pattern for input data""")

tf.app.flags.DEFINE_string('output', '../data/output/',
                           """Directory in which to write prediction output""")
tf.app.flags.DEFINE_bool('write_images', False,
                         """Save images of predictions""")

tf.app.flags.DEFINE_float('detect_thresh', 0.5,
                          """Threshold for rectangle detection""")
tf.app.flags.DEFINE_float('nms_thresh', 0.5,
                          """Threshold for non-maximal suppression""")
tf.app.flags.DEFINE_integer('pyramid_levels',1,
                            "Number of image pyramid levels")
FLAGS = tf.app.flags.FLAGS



def save_boxes_to_file(boxes, scores,output_base):
    """
    Save predicted bounding box coordinates and corresponding score
    to a text file

    In the text file, each line represents one box. If an arbitrary corner 
    point in boxes is (a_i, b_j), first the a_i's, then the b_j's, then an 
    empty string, then the score will be listed in the entry for that box.

    Parameters
       boxes       : a Nx4x2 numpy array of box vertices (N for number of boxes,
                       4 corners per box, and 2 coordinates per corner)
       scores      : a Nx1 numpy array of corresponding box scores
                      (N for the number of boxes)
       output_base : Path and basename for image-specific box output file 
                       ('.txt' extension is appended)
    """
    res_file = output_base+'.txt'

    print 'Saving',len(boxes),'boxes to',res_file
    
    with open(res_file, 'w') as f:
        for b in range(np.shape(scores)[0]):
            box = np.squeeze(boxes[b,:,:])
            f.write('{},{},{},{},{},{},{},{},"",{}\r\n'.format(
                box[0, 0], box[0, 1],
                box[1, 0], box[1, 1],
                box[2, 0], box[2, 1],
                box[3, 0], box[3, 1],
                scores[b]
            ))


def reconstruct_box(point, dist, theta):
    """
    Reconstructs a rotated rectangle from the corresponding point, distance to
    edges, and angle

    Parameters
      point: the (i, j) coordinate of the pixel as a numpy array
      dist : distances from the pixel to [top, left, bottom, and right] edges 
              as a numpy array
      theta: the angle of rotation (according to OpenCV), a scalar

    Returns
      box: a rotated rectangle defined by four (i,j) vertices in bottom-left, 
            bottom-right, top-right, and top-left order; a numpy array with
            shape [4,2]  (i.e., 4 corner points with 2 coordinates per point)

    Note
     i corresponds to row number, j corresponds to column number
    """
    
    left_trans = np.asarray([-math.cos(theta), -math.sin(theta)])*dist[0]
    bottom_trans = np.asarray([math.sin(theta), -math.cos(theta)])*dist[1]
    right_trans = np.asarray([math.cos(theta), math.sin(theta)])*dist[2]
    top_trans = np.asarray([-math.sin(theta), math.cos(theta)])*dist[3]
    
    v0_trans = left_trans + bottom_trans
    v1_trans = bottom_trans + right_trans
    v2_trans = right_trans + top_trans
    v3_trans = top_trans + left_trans

    point = point*4 # Compensate for downsampling in target map construction
    v0 = point + v0_trans
    v1 = point + v1_trans
    v2 = point + v2_trans
    v3 = point + v3_trans
    box = np.asarray([v1, v2, v3, v0])
    
    return box


def create_tile_set(image, tile_shape):
    """
    Create a set of tiles and their corresponding relative positions (shifts)
    A tile is a smaller section of the larger image.

    Parameters
       image      : image to tile, as numpy array, shape [H, W, 3]
       tile_shape : tuple of the tiles size as (width, height)

    Returns
       tiles  : list of tile images of shape tile_shape, 
                 where each tile image is a numpy array of shape 
                 (height, width, 3) and tile image is a portion of image
       shifts : list of (y,x) shifts necessary to translate points on each tile
                  to the original image
    """

    def tile_ticks( img_sz, tile_sz ):
        """ Calculate tile origin points and sizes.
            Tiles must overlap by at least the FLAGS.tile_overlap
        """
        ticks = list()
        sizes = list()
        
        pos = 0

        if tile_sz > img_sz:
            return [0], [img_sz]

        while (True):
            ticks.append( pos )
            sizes.append( tile_sz )
            
            next_pos = pos + tile_sz - FLAGS.tile_overlap

            if (next_pos + tile_sz) >= img_sz:
                trunc_tile_sz = img_sz - next_pos
                next_pos = img_sz - trunc_tile_sz

                ticks.append( next_pos )
                sizes.append( trunc_tile_sz )

                return ticks, sizes
            
            pos = next_pos

    # Main procedure
    tiles = list()
    shifts = list()
    tile_width = tile_shape[0]
    tile_height = tile_shape[1]
    im_width = len(image[0])
    im_height = len(image)

    y_tiles = tile_ticks( im_height, tile_height )
    x_tiles = tile_ticks( im_width, tile_width )

    print y_tiles
    print x_tiles
    
    # Loop over all tile (position, size) pairs
    for y,h in zip(y_tiles[0],y_tiles[1]):
        for x,w in zip(x_tiles[0],x_tiles[1]):
            tile = image[y:y+h, x:x+w]
            shift = (y,x)
    
            tiles.append( tile )
            shifts.append( shift )

    return tiles, shifts


def convert_geometry_to_boxes(score_map, geo_map, detect_thresh):
    """Convert the predicted geometry map into rotated rectangles

    Parameters:
       score_map     : the predicted score map, a numpy array of size
                         [batch_size, tile_size/4, tile_size/4, 1]
       geo_map       : the predicted geometry map, a numpy array of size
                         [batch_size, tile_size/4, tile_size/4, 5]
       detect_thresh : the minimum score of a bounding box for it to be 
                        counted as prediction, a scalar
    Returns:
       boxes : a Nx9 numpy array with the rectangle coordinates and scores;
                 each 1x9 row contains the four (i,j) pairs of vertices 
                 (in bottom-left, bottom-right, top-right, and top-left order) 
                 followed by the detection score
    """
    score_map = np.squeeze(score_map)
    geo_map = np.squeeze(geo_map)
    
    boxes = list()
    for i in xrange(len(score_map)):
        for j in xrange(len(score_map[0])):
            if (score_map[i, j] < detect_thresh):
                continue
            point = np.asarray([i, j])
            dist = geo_map[i, j, 0:4]
            theta = -geo_map[i, j, 4] # Negate to convert from OpenCV's ij to xy
            box = reconstruct_box(point, dist, theta)
            box = np.append(box, score_map[i, j])
            boxes.append(box)

    boxes = np.asarray(boxes)
    boxes = np.reshape(boxes, (len(boxes), 9))
    
    return boxes


def sort_by_row(boxes):
    """Sort the boxes by the row of their center points

    Parameters
       boxes: an Nx9 numpy array of rectangles and scores in ij order, as given 
                by convert_geometry_to_boxes

    Returns
       boxes: modification of input boxes so that the centers are sorted in
               increasing row order
    """
    # Calculate the row of the center of each rectangle
    # (geometrically, the mean of the vertices)
    center_rows = np.mean(boxes[:,[0,2,4,6]],axis=1) 
    # Extract the indices of the centers in sorted order
    center_rows_argsort = np.argsort(center_rows)

    return boxes[center_rows_argsort]


def predict(sess, image_file, pyramid_levels, input_images,
            f_score, f_geometry, tile_shape):
    """Use a restored model to detect text in the given image

    Parameters
       sess          : TensorFlow Session object
       image_file    : path of the image to run through the model, a string
       pyramid_levels: number of pyramid levels (decimations) before NMS
       input_images  : TensorFlow placeholder for image batch
       f_score       : TensorFlow tensor for model output (cf. model.outputs)
       f_geometry    : TensorFlow tensor for model output (cf. model.outputs)
       tile_shape    : tuple (width,height) of the tile size
    """

    image = cv2.imread(image_file)
    image = image[:, :, ::-1] # Convert from OpenCV's BGR to RGB
    boxes = np.zeros((0,9)) # Initialize array to hold resulting detections

    for level in xrange(pyramid_levels):
        if level != 0:
            image = cv2.resize( image, (0,0), fx=0.5, fy=0.5,
                                interpolation=cv2.INTER_CUBIC )

        image_tiles, shifts = create_tile_set(image, tile_shape)


        for i in xrange(len(image_tiles)):
            print 'predicting tile',i+1,'of',len(image_tiles)
            tile = image_tiles[i]
            shift = shifts[i]
            score, geometry = sess.run([f_score, f_geometry],
                                       feed_dict={input_images: [tile]})
            tile_boxes = convert_geometry_to_boxes(
                score_map=score,
                geo_map=geometry,
                detect_thresh=FLAGS.detect_thresh)

            if len(tile_boxes) != 0:
                # Shift boxes to global image coords from tile-specific coords
                shift = np.asarray([shift[0],shift[1],
                                    shift[0],shift[1],
                                    shift[0],shift[1],
                                    shift[0],shift[1],
                                    0])
                tile_boxes = tile_boxes[:,:]+shift
                # Resize tile boxes to global image coords from pyramid-level
                tile_boxes[:,:-1] *= (2**level)
                boxes = np.concatenate((boxes, tile_boxes), axis=0)
    print 'LANMS...'
    boxes = sort_by_row(boxes) # still ij
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), FLAGS.nms_thresh)

    scores = boxes[:,-1]
    boxes = boxes[:, :8].reshape(-1, 4, 2)
    boxes = np.flip(boxes, axis=2) #IMPORTANT ij-xy conversion

    output_base = os.path.join(FLAGS.output,
                            os.path.splitext(
                                os.path.basename( image_file ))[0] )
    print 'writing output'
    if boxes is not None:
        save_boxes_to_file(boxes, scores, output_base)
        
    if FLAGS.write_images:
        visualize.save_image( image, boxes, output_base)
    

def restore_model(sess):
    """Restore model parameters from latest checkpoint within a given session

    Parameters:
       sess: the TensorFlow Session object
    """
    saver = tf.train.Saver()
    ckpt_state = tf.train.get_checkpoint_state(FLAGS.model)
    model_path = os.path.join(FLAGS.model,
                              os.path.basename(ckpt_state.model_checkpoint_path))
    saver.restore(sess, model_path)

    
def main(argv=None):
    """
    Establishes a TensorFlow session, restores model, and runs detector
    on image(s), writing bounding box information to a text file.
    """
    image_filenames = data_tools.get_filenames(
         FLAGS.image_path, str.split(FLAGS.filename_pattern,','), 'tiff')

    if not image_filenames:
        print "No matching images. Exiting..."
        return
    
    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3],
                                      name='input_images')
        f_score, f_geometry = model.outputs(input_images, is_training=False)

        with tf.Session() as sess:

            restore_model(sess)

            for image_file in image_filenames:
                predict(sess, image_file, FLAGS.pyramid_levels,
                        input_images, f_score, f_geometry,
                        (FLAGS.tile_size,FLAGS.tile_size))


if __name__ == '__main__':
    tf.app.run()
