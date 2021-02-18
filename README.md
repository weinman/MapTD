# MapTD: Map Text Detector

## Overview

MapTD is a convolutional neural network model for text detection
written in Tensorflow.  It is engineered for operating on mixed
text/graphics documents, such as maps.

The code is a derivative of
@[argman](https://github.com/argman)'s 
[implementation](https://github.com/argman/EAST) of 
[EAST: An Efficient and Accurate Scene Text Detector](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_EAST_An_Efficient_CVPR_2017_paper.pdf)
by Zhou et al.

Like argman's EAST code, the network backbone is Resnet-50, with a dice
loss (whereas the EAST paper uses PVANet with balanced cross-entropy loss).

This model uses a semantically-grounded orientation for the
rectangles, so that it learns to produce rectangle sides in a specific
order (i.e., text baseline, right, text top, left).

## Installation

The package requires Tensorflow >=1.12 and the other Python
dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Model Structure

The MapTD network consists of four distinct parts.

### Feature Extraction and Merging

First, the algorithm extracts four internal convolutional feature maps
from the ResNet-50 backbone---which are 1/32, 1/16, 1/8, and 1/4 of
the size of the input image. Then, these feature maps are merged in
the following way. Sequentially, each feature map is concatenated with
a version of the previous merged feature maps that has been doubled in
size through bilinear upsampling. The result is bottlenecked with a
1x1 convolution (decreasing the number of channels) and a 3x3
feature-fusing convolution. Finally, once all ResNet-50 four feature
maps have been encorporated into the merged tensor, a 3x3 convolution
is applied to produce the final feature map for the output layers.

### Output Detection Layers

The feature map for the output layers is manipulated to produce three
independent output layers with the following values at each output
location:

1. A score indicating the probability that text is present for
   each point in the map (fully connected layer with a sigmoid).
2. Box geometry specifying the distance to the edges of a rotated,
   rectangular bounding box (fully connected layer with a sigmoid for
   stability, scaled by a constant to the fixed maximal ).
3. Angle of rotation for the bounding box (fully connected layer
   with arctan).

### Loss Functions

MapTD associates a loss function with each of the three output
layers. The score map is learned with Dice loss, an IOU loss governs
the box geometry map, and rotation angle loss is defined to be cosine
loss.

### Suppression

Finally, to minimize redundant boundary boxes with relative
efficiency, MapTD uses Locality-Aware Non-Maximal Suppression which
iteratively merges rectangles, weighted by average score row by row
before applying a standard Non-Maximal Suppresion algorithm.


## Locality-Aware Non-Maximal Suppression

This code requires the "Locality-Aware Non-Maximal Suppression"
(LANMS) of the original paper's author, as provided by
[argman/EAST](https://github.com/argman/EAST). On Linux, it is
automatically when running the code, but to explicitly compile:

```bash
cd maptd
make -C lanms
```

To compile on Windows, see 
[EAST issue #120](https://github.com/argman/EAST/issues/120)

## Usage

### Training

Training is accomplished in `train.py`. Simply point the command-line
flags to the correct training data (data should be in JSON format with
"points" and optional "text" label) and run the script.

#### Checkpoints

[Pre-trained model checkpoints](http://hdl.handle.net/11084/23329) at
DOI:[11084/23329](http://hdl.handle.net/11084/23329) are used to
produce results in the following paper:

> Weinman, J. et al. (2019) Deep Neural Networks for Text Detection
> and Recognition in Historical Maps. In Proc. ICDAR.

### Testing

Using a train model can involve two steps: running the learned model
to produce predictions and running a separate evaluation against
ground truth to calculate resulting statistics.

### Predict

To predict, run `predict.py` setting any needed flags, including the
path to the store model and input image locations. It produces a text
file for each image storing the detected rectangles and (optionally) a
visualization of the rectangles.

### Evaluate

Run `evaluate.py` to calculate precision, recall, f-score, and average
precision of the predictions.

## Notes

* When documentation references (i,j) coordinates, 'i' corresponds to
row number and 'j' corresponds to column number. In the case of (x,y)
coordinates, 'x' refers to column number and 'y' refers to row number.

## Citing this work

Please cite the following [paper](https://weinman.cs.grinnell.edu/pubs/weinman19deep.pdf) if you use this code in your own research work:

```text
@inproceedings{ weinman19deep,
    author = {Jerod Weinman and Ziwen Chen and Ben Gafford and Nathan Gifford and Abyaya Lamsal and Liam Niehus-Staab},
    title = {Deep Neural Networks for Text Detection and Recognition in Historical Maps},
    booktitle = {Proc. IAPR International Conference on Document Analysis and Recognition},
    month = {Sep.},
    year = {2019},
    location = {Sydney, Australia},
    doi = {10.1109/ICDAR.2019.00149}
} 
```

## Acknowledgements

Contributions are documented in (AUTHORS.md) and (CREDITS).

This work was supported in part by the National Science Foundation
under grant Grant Number
[1526350](http://www.nsf.gov/awardsearch/showAward.do?AwardNumber=1526350).
