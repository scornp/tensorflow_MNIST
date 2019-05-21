#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
#import cv2
import numpy as np
from PIL import Image
from openvino.inference_engine import IENetwork, IEPlugin
import matplotlib.pyplot as plt

IMAGE                   = '1xxB.png'
ROOT_PATH               = '.'
IMAGE_PATH              = ROOT_PATH +'/../../test_data/' + IMAGE
IMAGE_DIM               = ( 28, 28 )


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.",  required=True, type=str)

    parser.add_argument("-l", "--cpu_extension", help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels impl.", type=str, default=None)

    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str,default=None)
    parser.add_argument("-d", "--device", help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample " "will look for a suitable plugin for device specified (CPU by default)", default="CPU", type=str)
    parser.add_argument("-nt", "--number_top", help="Number of top results", default=10, type=int)

    return parser


def main():
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)
    # Read IR
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)
    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    # Read and pre-process input image

#-rnp n, c, h, w = net.inputs[input_blob]
    h, w = net.inputs[input_blob]
    #### image = cv2.imread(args.input)
#####
    imgOriginal = Image.open(IMAGE_PATH)
    img = imgOriginal
    img = img.resize(IMAGE_DIM)
    print(img)
    im2arr = np.array(img).reshape([784, 1])
    im2arr = im2arr.astype( np.float32 )
    image = im2arr

#####

#-rnp    image = cv2.resize(image, (w, h))
#####   image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
#-rnp    image = image.reshape((n, c, h, w))
#-rnp    image = image.reshape((h, w))
    # Load network to the plugin
    exec_net = plugin.load(network=net)
    del net
    # Start sync inference
    res = exec_net.infer(inputs={input_blob: image})
    top_ind = np.argsort(res[out_blob], axis=1)[0, -args.number_top:][::-1]
    for i in top_ind:
        print("%f #%d" % (res[out_blob][0, i], i))
    del exec_net
    del plugin

    fig = plt.figure()
    a=fig.add_subplot(1,3,1)

    x = range( 0, 10)
    y = np.zeros(10)
    for i in top_ind:
        y[i] = res[out_blob][0, i]
    width = 0.35

    plt.bar(x, y, width)
    plt.xlabel('digit')
    plt.ylabel('prediction')

    a=fig.add_subplot(1,3,2)
    plt.imshow(imgOriginal)

    a=fig.add_subplot(1,3,3)
    plt.imshow(img, cmap=plt.get_cmap('gray_r'))
    plt.show()


if __name__ == '__main__':
    sys.exit(main() or 0)
