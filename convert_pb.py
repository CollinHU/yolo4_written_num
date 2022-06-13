#================================================================
#
#   File name   : Convert_to_pb.py
#   Author      : PyLessons
#   Created date: 2020-08-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : used to freeze tf model to .pb model
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import sys
import tensorflow as tf
from core.yolov4 import YOLOv4
from tensorflow.keras.layers import  Input

TRAIN_CLASSES = "mnist/mnist.names"
TRAIN_YOLO_TINY = False
YOLO_TYPE = "yolov4"
 
STRIDES            =           [8, 16, 32]
ANCHORS            =           [[[12,  16], [19,   36], [40,   28]],
                               [[36,  75], [76,   55], [72,  146]],
                               [[142,110], [192, 243], [459, 401]]]

STRIDES         =    np.array(STRIDES)
ANCHORS         =   (np.array(ANCHORS).T/STRIDES).T

def decode_tf(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.reshape(conv_output,
                             (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS),
                                                                          axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
              STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_prob = pred_conf * pred_prob
    pred_prob = tf.reshape(pred_prob, (batch_size, -1, NUM_CLASS))
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))

    return pred_xywh, pred_prob
def decode(conv_output, output_size, NUM_CLASS, i, XYSCALE=[1,1,1], FRAMEWORK='tf'):
    if FRAMEWORK == 'trt':
        return decode_trt(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)
    elif FRAMEWORK == 'tflite':
        return decode_tflite(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)
    else:
        return decode_tf(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)

def read_class_names(class_file_name):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names
def Create_Yolo(input_size=416, channels=3, training=False, CLASSES=TRAIN_CLASSES):
    NUM_CLASS = len(read_class_names(CLASSES))
    input_layer  = Input([input_size, input_size, channels])

    if TRAIN_YOLO_TINY:
        if YOLO_TYPE == "yolov4":
            conv_tensors = YOLOv4_tiny(input_layer, NUM_CLASS)
    else:
        if YOLO_TYPE == "yolov4":
            conv_tensors = YOLOv4(input_layer, NUM_CLASS)
       
    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor,input_size, NUM_CLASS, i)
        if training: output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    Yolo = tf.keras.Model(input_layer, output_tensors)
    return Yolo



yolo = Create_Yolo(input_size=224, CLASSES=TRAIN_CLASSES)
Darknet_weights="checkpoints/yolo224/yolov4"
yolo.load_weights(Darknet_weights) # use custom weights
yolo.summary()
yolo.save(f'./checkpoints/my_model')

print(f"model saves to /checkpoints/my_model")
