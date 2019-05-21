#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ./binarize.py -i convert_image.png -o result_bin.png --threshold 200
"""Binarize (make it black and white) an image with Python."""

from PIL import Image
from scipy.misc import imsave
import numpy as np

IMAGE_DIM = (28, 28)


def binarize_image(img_path, target_path, threshold):
    """Binarize an image."""
    image_file = Image.open(img_path)
    image_file = image_file.resize(IMAGE_DIM)

 #   image = image_file.convert('L')  # convert image to monochrome
    image = np.array(image_file)
    img = np.zeros(IMAGE_DIM)
   # img = img.resize(IMAGE_DIM)
    for i in range(28):
        for j in range(28):
          img[i][j] = image[i][j][0]#.astype(np.uint8)
 #   print(image)
#    for i in range(len(image)):
#        for j in range(len(image[0])):
    for i in range(28):
        for j in range(28):
           str = ''
           str = "{} {} {}".format(i, j, image[i][j][0])
           print(str) 
        #  print( str + image[i][j])
         #  str = ''
          # str = str + String(i)
         #  print( str + image[i][j])
            
      #  print('\n')
  #  image = binarize_array(image, threshold)
    imsave(target_path, img)


def binarize_array(numpy_array, threshold=200):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input",
                        dest="input",
                        help="read this file",
                        metavar="FILE",
                        required=True)
    parser.add_argument("-o", "--output",
                        dest="output",
                        help="write binarized file hre",
                        metavar="FILE",
                        required=True)
    parser.add_argument("--threshold",
                        dest="threshold",
                        default=200,
                        type=int,
                        help="Threshold when to show white")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    binarize_image(args.input, args.output, args.threshold)
