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

# model.py -- Constructs the graph representing the MapTD network model and 
#   defines the loss functions for training

import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim
from nets import resnet_v1


def unpool(inputs,new_size=None):
    """
    Double the feature map patial size via bilinear upsampling

    Parameters
     inputs: feature map of rank 4 to be upsampled (changes in spatial size)
     new_size: Two-tuple (width, height) of new size for outputs. If not given, 
               default behavior doubles the size.
    Returns
     outputs: upsized rank 4 tensor. If inputs shape is [a, b, c, d], outputs
               shape is (a, new_size[0], new_size[1], d)
    """
    if not new_size:
        new_size = [tf.shape(inputs)[1]*2, tf.shape(inputs)[2]*2]
        
    return tf.image.resize_bilinear( inputs,
                                     size=new_size)


def outputs(images, l2_weight_decay=1e-5, is_training=True):
    """
    Generate the output layer tensors for a batch of input image tiles

    Parameters
      images         : tensor of inputs [batch_size, tile_size, tile_size, 3]
      l2_weight_decay: scalar for the net's weight L2 regularizer
      is_training    : boolean value indicating the model graph is to be trained

    Returns
      F_score   : tensor [batch_size, tile_size/4, tile_size/4, 1] representing
                    the score for text at each point
      F_geometry: tensor [batch_size, tile_size/4, tile_size/4, 5]. The first
                    four channels are the distances from each location to edges
                    of a rectangular bounding box in (top, left, bottom, right
                    order). The last channel is the angle of rotation for the
                    bounding box.

    Notes
     Uses a tf slim implemention of resnet from an early version of tensorflow.
    """
    
    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=l2_weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images,
                                                    is_training=is_training,
                                                    scope='resnet_v1_50')
        
    with tf.variable_scope('feature_fusion', values=[end_points.values]):

        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        
        with slim.arg_scope(
                [slim.conv2d],
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                weights_regularizer=slim.l2_regularizer(l2_weight_decay)):

            f = [ end_points['pool5'], end_points['pool4'],
                  end_points['pool3'], end_points['pool2'] ]

            #for i in range(4):
            #    print('Shape of f_{} {}'.format(i, f[i].shape))
            
            g = [None, None, None, None]
            h = [None, None, None, None]
            num_outputs = [None, 128, 64, 32]
            
            for i in range(4):
                
                if i == 0:
                    h[i] = f[i]
                else:
                    c1_1 = slim.conv2d( tf.concat( [ g[i-1], f[i] ], axis=-1),
                                        num_outputs[i], 1)
                    
                    h[i] = slim.conv2d( c1_1, num_outputs[i], 3)
                if i <= 2:
                    new_sz = tf.shape(f[i+1])
                    g[i] = unpool(h[i],(new_sz[1],new_sz[2]))
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                #print('Shape of h_{} {}, g_{} {}'.format
                #      (i, h[i].shape, i, g[i].shape))

            F_score = slim.conv2d( g[3], 1, 1,
                                   activation_fn=tf.nn.sigmoid,
                                   normalizer_fn=None)

            # From argman/EAST: using a different regression setup,
            # with a sigmoid/atan to limit the regression range for the 
            # geometry/angle, followed by a scale factor adjustment
            
            #geo_scale = tf.Variable(1.0,dtype=tf.float32)

            #if is_training: # Set to train tile size
            #    geo_scale.assign(tf.cast( tf.shape(images)[1], tf.float32))
            # TODO: Fix geo_scale above to work in tower context
            geo_scale = 512.0
            # 4 channel of axis aligned bbox and 1 channel rotation angle    
            geo_map = geo_scale * slim.conv2d( g[3], 4, 1,
                                                 activation_fn=tf.nn.sigmoid,
                                                 normalizer_fn=None) 
            # angle is between (-pi, pi)
            angle_map = 2 * slim.conv2d( g[3], 1, 1,
                                          activation_fn=tf.atan,
                                          normalizer_fn=None) 
            F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    return F_score, F_geometry


def dice_coefficient( y_true_cls, y_pred_cls, training_mask):
    """
    Compute Dice loss from Sorensen-Dice coefficient. See Eq. (1) Weinman et al.
    (ICDAR 2019) and Milletari et al. (3DV 2016).

    Parameters
      y_true_cls   : binary ground truth score map (1==text, 0==non-text), 
                       [batch_size, tile_size/4, tile_size/4, 1]
      y_pred_cls   : predicted score map in range (0,1), same size as y_true_cls
      training_mask: binary tensor to indicate which locations should be included 
                       in the calculation, same size as y_true_cls
    Returns
      loss: scalar tensor between 0 and 1
    """
    eps = 1e-5 # added for numerical stability
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum( tf.square(y_true_cls) * training_mask) + \
            tf.reduce_sum( tf.square(y_pred_cls) * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss, family='train/losses')
    return loss


def loss(y_true_cls, y_pred_cls,
         y_true_geo, y_pred_geo,
         training_mask):
    '''
    Compute total loss as the weighted sum of score loss (given by a
    Dice loss), rbox loss (defined as an IoU loss), and angle loss
    (i.e., cosine loss).  See Eq. (6) in Weinman et al. (ICDAR 2019).

    Parameters
     y_true_cls   : binary ground truth score map (1==text, 0==non-text), 
                    [batch_size,tile_size/4,tile_size/4, 1]
     y_pred_cls   : predicted score map in range (0,1), same size as y_true_cls
     y_true_geo   : ground truth box geometry map with shape 
                      [batch_size,tile_size/4,tile_size/4, 5]
     y_pred_geo   : predicted box geometry map, same size as y_true_geo
     training_mask: binary tensor to indicate which locations should be included 
                      in loss the calculations, same size as y_true_cls

    Returns
     total_loss: a scalar

    '''
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(
        value=y_true_geo,
        num_or_size_splits=5,
        axis=3)
    
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(
        value=y_pred_geo,
        num_or_size_splits=5,
        axis=3)
    
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    
    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    
    L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0))
    L_theta = 1 - tf.cos(theta_pred - theta_gt)
    
    tf.summary.scalar('geometry_AABB',
                      tf.reduce_mean(L_AABB * y_true_cls * training_mask),
                      family='train/losses')
    tf.summary.scalar('geometry_theta',
                      tf.reduce_mean(L_theta * y_true_cls * training_mask),
                      family='train/losses')
    
    L_g = L_AABB + 20 * L_theta

    total_loss = tf.reduce_mean(L_g * y_true_cls * training_mask) \
                 + classification_loss

    return total_loss
