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

import tensorflow as tf
import os
import data_tools
import stats


tf.app.flags.DEFINE_string('gt_path','../data/test',
                           """Base directory for ground truth data""")
tf.app.flags.DEFINE_string('pred_path','../data/predict',
                           """Base directory for predicted output data""")
tf.app.flags.DEFINE_string('filename_pattern','*',
                           """File pattern for data""")
tf.app.flags.DEFINE_float('iou_thresh',0.5,
                          """Intersection-over-Union threshold for match""")
tf.app.flags.DEFINE_float('score_thresh',None,
                          """Score threshold for predictions""")
tf.app.flags.DEFINE_boolean('match_labels',False,
                            """Whether to require labels to match""")
tf.app.flags.DEFINE_string('save_result',None,
                          """JSON file in which to save results in pred_path""")


FLAGS = tf.app.flags.FLAGS

def threshold_predictions(polys,labels,scores):

  t_polys = list()
  t_labels = list()
  t_scores = list()

  for (poly,label,score) in zip(polys,labels,scores):
    if score > FLAGS.score_thresh:
      t_polys.append(poly)
      t_labels.append(label)
      t_scores.append(score)
  return t_polys,t_labels,t_scores


def main(argv=None):
    """Loads up ground truth and prediction files, calculates and
       prints statistics
    """

    # Load file lists
    prediction_files = data_tools.get_filenames(
      FLAGS.pred_path,
      str.split(FLAGS.filename_pattern,','),
      'txt')
    
    ground_truth_files = data_tools.get_paired_filenames(
      prediction_files, FLAGS.gt_path, 'json' )

    assert len(ground_truth_files) == len(prediction_files)

    # Load files contents and package for stats evaluation
    predictions = {}
    ground_truths = {}
    
    for pred_file,truth_file in zip(prediction_files,ground_truth_files):

      base = os.path.splitext(os.path.basename(pred_file))[0]

      [_,gt_polys,gt_labels] = data_tools.parse_boxes_from_json( truth_file )
      [_,polys,labels,scores] = data_tools.parse_boxes_from_text( pred_file )

      if FLAGS.score_thresh: # Filter predictions if necessary
        polys,labels,scores = threshold_predictions(
          polys, labels, scores)

      predictions[base] =  { 'polygons' : polys,
                             'labels'   : labels,
                             'scores'   : scores }
      ground_truths[base] = {'polygons' : gt_polys,
                             'labels'   : gt_labels }

    # Calculate statistics on predictions for ground truths
    sample_stats,total_stats = stats.evaluate_predictions(
      ground_truths,
      predictions,
      match_labels=FLAGS.match_labels,
      iou_match_thresh=FLAGS.iou_thresh)

    # Display save the results
    print sample_stats
    print total_stats
    
    if FLAGS.save_result:
      import json
      with open(os.path.join(FLAGS.pred_path,FLAGS.save_result+'.json'),'w') \
           as fd:
        json.dump({'individual': sample_stats, 'overall': total_stats}, fd,
                  indent=4)

    
if __name__ == "__main__":
    tf.app.run()
     
