# -*- coding: utf-8 -*-
# encoding=utf8
#
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

#  stats.py -- Utilities for evaluating prediction statitics on ground truth

import tensorflow as tf
import numpy as np
from shapely.geometry.polygon import Polygon

import data_tools

def polygon_union(poly1, poly2):
  """Calculate the area of the union of two polygons

  Parameters
     poly1 : a polygon (shapely.geometry.polygon.Polygon object)
     poly2 : a polygon (shapely.geometry.polygon.Polygon object)

  Returns
     area_inter: the area of union of the two given polygons, a scalar
  """
  area1 = poly1.area;
  area2 = poly2.area;
  area_inter = polygon_intersection(poly1, poly2)
  return area1 + area2 - area_inter;


def polygon_intersection(poly1, poly2):
  """Calculate the area of the intersection between two polygons

  Parameters
     poly1 : a polygon (shapely.geometry.polygon.Polygon object)
     poly2 : a polygon (shapely.geometry.polygon.Polygon object)

  Returns
     area_inter: the area of intersection of the two givenpolygons, a scalar
  """
  if not poly1.intersects(poly2):
    return 0

  try:
    area_inter = poly1.intersection(poly2).area;
  except shapely.geos.TopologicalError:
    print('shapely.geos.TopologicalError occurred, intersection set to 0')
    area_inter = 0
  return area_inter
    

def polygon_iou(poly1, poly2):
  """Calculate the area of Intersection-over-area of Union between two polygons

  Parameters
     poly1 : a polygon (shapely.geometry.polygon.Polygon object)
     poly2 : a polygon (shapely.geometry.polygon.Polygon object)

  Returns
     iou : the IOU of the two polygons, a scalar
  """
  if not poly1.intersects(poly2):
    iou = 0
  else:
    try:
      union_area = polygon_union(poly1, poly2)
      inter_area = polygon_intersection(poly1, poly2)
      iou = float(inter_area) / union_area
    except shapely.geos.TopologicalError:
      print('shapely.geos.TopologicalError occurred, iou set to 0')
      iou = 0

  return iou

        
def compute_ap(scores, matches, num_truths):
  """Compute average precision metric over a list of scored matches

  Parameters:
    scores     :  Nx1 numpy array of prediction scores
    matches    :  boolean  Nx1 numpy array indicating correct predictions
    num_truths :  Number of ground truths (denominator in precision), a scalar
  Returns:
    avg_precision : Total AP metric for the given set of predictions, a scalar
  """
  correct = 0
  avg_precision = 0

  if len(scores)>0:
    # Put prediction scores and corresponding matches in decreasing sorted order
    scores = np.asarray(scores)
    matches = np.asarray(matches)

    sorted_ind = np.argsort(-scores)
    scores     = scores[sorted_ind]
    matches    = matches[sorted_ind]
    
    for n in xrange(len(scores)):
      if matches[n]:
        correct += 1
        avg_precision += float(correct)/(n + 1)

    if num_truths>0:
      avg_precision /= num_truths
            
  return avg_precision

  
def evaluate_predictions(ground_truths, predictions, 
                         match_labels=False,
                         iou_match_thresh=0.5,
                         dont_care_overlap_thresh=0.5 ):
  
  """Evalute statistics of predictions relative to ground truth rectangles. 

     Parameters
       ground_truths : Dictionary of ground truths, each key representing an
                         arbitrary name for a sample and each value being a 
                         dictionary with keys 'polygons' and 'labels', each 
                         of those values being matched lists of Shapely 
                         polygons and strings. For example,
                          {keyA : {'polygons' : [poly1,poly2,...], 
                                   'labels'   : [string1,string2,...]},
                           ... }
       predictions   : Matching dictionary of predictions, with additional 
                         sub-key 'scores' containing a NumPy array of scores 
                         corresponding to each polygon.
       dont_care_overlap_thresh : Minimum ratio of |det⋂gt|/|det| for counting a
                                    matched rectangle as a don't care when gt 
                                    is marked don't care

     Returns
       stats       : Dictionary with the same keys as predictions, each value 
                       being a per-sample statistics dictionary with keys 
                       num_predictions, num_ground_truths, num_correct, 
                       precision, recall, fscore, ap.
       total_stats : Total statistics dictionary for pooled results with the same
                       keys as dictionaries in stats
  """

  # BEGIN Local Functions
  def find_dont_cares(labels):
    """Return indices of labels indicating a don't care polygon. 
       Empty if match_labels == False.
    """
    if match_labels:
      return np.where(np.array(labels)==None)[0].tolist()
    else:
      return []

    
  def match_dont_cares(dont_cares, true_polys, det_polys):
    """Return indices of det_polys that match an entry in  
       true_polys[dont_cares]."""

    matches = list()

    if len(dont_cares)==0:
      return matches
    
    for n in xrange(len(det_polys)):
      det_poly = det_polys[n]
      for d in dont_cares:
        dont_care_poly = true_polys[d]
        intersected_area = polygon_intersection(dont_care_poly,det_poly)
        overlap_ratio = 0 if det_poly.area == 0 else \
                        intersected_area/det_poly.area

        if (overlap_ratio > dont_care_overlap_thresh):
          matches.append( n )
          break
    return matches

  
  def get_all_ious(true_polys, det_polys):
    """Return the iou scores of all pairs of polygons"""
    num_gt   = len(true_polys)
    num_det = len(det_polys)
    
    iou_mat = np.empty( [num_gt, num_det] )
    for g in xrange(num_gt):
      for d in xrange(num_det): 
        iou_mat[g,d] = polygon_iou(true_polys[g], det_polys[d])
    return iou_mat


  def get_matches(true_polys, det_polys,
                  gt_dont_cares, det_dont_cares):
    """Returns a list of indices to det_polys for which a match is found"""

    num_gt   = len(true_polys)
    num_det = len(det_polys)

    matched = list()
    
    if num_gt==0 or num_det==0:
      return matched
    
    iou_mat = get_all_ious( true_polys, det_polys )

    det_matched = np.zeros(num_det, dtype=np.bool_) # Matches initially false

    for g in xrange(num_gt):
      for d in range(num_det):
        # Skip the pairing if we've already matched the detection or
        # either polygon is a don't care
        if not det_matched[d] and \
          (g not in gt_dont_cares) and (d not in det_dont_cares):
          if iou_mat[g,d] > iou_match_thresh: # Match if above threshold
            det_matched[d] = True
            # correct ≡  match_labels ⇒ (gt_labels==predicted_labels)
            correct = not match_labels or \
                      ground_truth['labels'][g] == prediction['labels'][d]
            if correct:
              matched.append(d)
            break # match found: stop searching over detections
    return matched

  
  def get_stats(scores, matches, num_gt, num_det, num_correct):
    """Calculate dict of PRF+AP stats for the matches
       Parameters:
         scores      :  Nx1 array of prediction scores
         matches     :  Nx1 boolean array, whether each prediction is correct
         num_gt      :  Total ground truth rectangles (excluding don't cares)
         num_det     :  Total predicted rectangles (excluding don't cares)
         num_correct :  Total number of correct predictions
    """
    if num_gt == 0:
      recall = 1.0
      precision = 0.0 if num_det > 0 else 1.0
      sample_ap = precision
    else:
      recall = num_correct / num_gt
      precision = 0.0 if num_det==0 else num_correct / num_det
      sample_ap = compute_ap(scores, matches, num_gt )
      
    if (precision + recall)==0:
      fscore = 0
    else:
      fscore = 2.0 * precision * recall / (precision + recall)

    results = { 'num_predictions'  : int(num_det),
                'num_ground_truths': int(num_gt),
                'num_correct'      : int(num_correct),
                'precision'        : precision,
                'recall'           : recall,
                'fscore'           : fscore,
                'ap'               : sample_ap }
    
    return results

  def process_sample(ground_truth, prediction):
    """Main function for calculating statistics on a single sample"""
    scores = []
    matches = [] # Booleans indicating whether the detection was correct
    
    gt_dont_cares  = find_dont_cares( ground_truth['labels'] )
    det_dont_cares = match_dont_cares( gt_dont_cares, ground_truth['polygons'],
                                       prediction['polygons'])
    matched = get_matches( ground_truth['polygons'], prediction['polygons'],
                           gt_dont_cares, det_dont_cares )

    num_gt = len(ground_truth['polygons'])
    num_det = len(prediction['polygons'])
      
    num_gt_care = num_gt - len(gt_dont_cares)
    num_det_care = num_det - len(det_dont_cares)
    num_correct = float(len(matched))
    
    for d in xrange(num_det):
      if d not in det_dont_cares: # exclude don't care detections
        scores.append(prediction['scores'][d])
        matches.append(d in matched)
        
    results = { 'scores'      : scores,
                'matches'     : matches,
                'num_gt'      : num_gt_care,
                'num_det'     : num_det_care,
                'num_correct' : num_correct }
    
    stats = get_stats(scores, matches,
                      num_gt_care, num_det_care, num_correct)
    
    return results, stats
  # END Local Functions
  
  # predictions and ground truths must have the same entries
  assert( set(predictions.keys()) == set(ground_truths.keys()))
  
  stats = {} # Per-sample results
  
  total_num_correct = 0
  total_num_gt = 0
  total_num_det = 0
  
  total_scores = [];
  total_matches = []; # Booleans indicating whether the detection was correct

  for sample in ground_truths:

    ground_truth = ground_truths[sample]
    prediction = predictions[sample]

    [results,sample_stats] = process_sample(ground_truth,prediction)
    
    stats[sample] = sample_stats
    
    total_scores += results['scores']
    total_matches += results['matches']

    total_num_correct += results['num_correct']
    total_num_gt += results['num_gt']
    total_num_det += results['num_det']


  total_stats = get_stats(total_scores, total_matches,
                          total_num_gt, total_num_det,
                          total_num_correct)
  total_stats['iou_threshold'] = iou_match_thresh

  return stats, total_stats
