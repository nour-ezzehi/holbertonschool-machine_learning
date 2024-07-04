#!/usr/bin/env python3
""" 2. Filter Boxes """


import tensorflow as tf
import numpy as np


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

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """ Returns a tuple of (boxes, box_confidences, box_class_probs) """
        boxes = list()
        box_confidences = list()
        box_class_probs = list()

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes = output.shape[:3]

            box_xy = self.sigmoid(output[..., :2])
            box_wh = np.exp(output[..., 2:4]) * self.anchors[i]
            box_confidence = self.sigmoid(output[..., 4:5])
            box_class_prob = self.sigmoid(output[..., 5:])

            col = np.tile(np.arange(0, grid_width),
                          grid_height).reshape(-1, grid_width)
            row = np.tile(np.arange(0, grid_height).reshape(-1, 1), grid_width)
            col = col.reshape(
                grid_height, grid_width, 1, 1).repeat(
                anchor_boxes, axis=2)
            row = row.reshape(
                grid_height, grid_width, 1, 1).repeat(
                anchor_boxes, axis=2)
            grid = np.concatenate((col, row), axis=-1)

            box_xy = (box_xy + grid) / [grid_width, grid_height]
            box_wh = box_wh / [self.model.input.shape[1],
                               self.model.input.shape[2]]

            box_x1y1 = box_xy - (box_wh / 2)
            box_x2y2 = box_xy + (box_wh / 2)
            boxes_per_output = np.concatenate((box_x1y1, box_x2y2), axis=-1)

            boxes_per_output[..., 0] *= image_width
            boxes_per_output[..., 1] *= image_height
            boxes_per_output[..., 2] *= image_width
            boxes_per_output[..., 3] *= image_height

            boxes.append(boxes_per_output)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return (boxes, box_confidences, box_class_probs)


def filter_boxes(self, boxes, box_confidences, box_class_probs):
    """ Determines filtered bounding boxes from processed outputs """

    filtered_boxes = list()
    box_classes = list()
    box_scores = list()

    for i, b in enumerate(boxes):
        box_conf = box_confidences[i]
        box_class_prob = box_class_probs[i]
        box_score = box_conf * box_class_prob
        box_class = np.argmax(box_score, axis=-1)
        box_class_score = np.max(box_score, axis=-1)

        index = np.where(box_class_score >= self.class_t)
        filtered_boxes.append(b[index])
        box_classes.append(box_class[index])
        box_scores.append(box_class_score[index])

    filtered_boxes = np.concatenate(filtered_boxes)
    box_classes = np.concatenate(box_classes)
    box_scores = np.concatenate(box_scores)
    return (filtered_boxes, box_classes, box_scores)
