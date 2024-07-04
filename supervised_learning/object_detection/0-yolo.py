#!/usr/bin/env python3
""" 0. Initialize Yolo """


import tensorflow as tf


class Yolo:
    """ uses the Yolo v3 algorithm to perform object detection """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ YOLO class constructor """

        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as file:
            self.class_names = file.read().splitlines()

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
