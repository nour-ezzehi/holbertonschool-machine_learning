#!/usr/bin/env python3
""" 3. Non-max Suppression """


import tensorflow.keras as K
import numpy as np


class Yolo:
    """In this task, we use the yolo.h5 file."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize class constructor"""
        self.model = K.models.load_model(model_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

        with open(classes_path, 'r') as f:
            classes = f.read().strip().split('\n')
        self.class_names = classes

    def sigmoid(self, x):
        """Apply sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """Process and normalize the output of the YoloV3 model."""
        boxes = []
        box_confidences = []
        box_class_probs = []
        img_h, img_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, nb_box, _ = output.shape
            box_conf = self.sigmoid(output[..., 4:5])
            box_prob = self.sigmoid(output[..., 5:])
            box_confidences.append(box_conf)
            box_class_probs.append(box_prob)

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            c_x = np.arange(grid_w).reshape(1, grid_w, 1)
            c_x = np.tile(c_x, [grid_h, 1, nb_box])
            c_y = np.arange(grid_h).reshape(grid_h, 1, 1)
            c_y = np.tile(c_y, [1, grid_w, nb_box])

            p_w = self.anchors[i, :, 0].reshape(1, 1, nb_box)
            p_h = self.anchors[i, :, 1].reshape(1, 1, nb_box)

            b_x = (self.sigmoid(t_x) + c_x) / grid_w
            b_y = (self.sigmoid(t_y) + c_y) / grid_h
            b_w = (np.exp(t_w) * p_w) / self.model.input.shape[1]
            b_h = (np.exp(t_h) * p_h) / self.model.input.shape[2]

            x1 = (b_x - b_w / 2) * img_w
            y1 = (b_y - b_h / 2) * img_h
            x2 = (b_x + b_w / 2) * img_w
            y2 = (b_y + b_h / 2) * img_h

            box = np.zeros((grid_h, grid_w, nb_box, 4))
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2
            boxes.append(box)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filters boxes based on their box scores"""
        filtered_boxes = []
        filtered_scores = []
        filtered_classes = []

        for i in range(len(boxes)):
            box_scores = box_confidences[i] * box_class_probs[i]
            box_classes = np.argmax(box_scores, axis=-1)
            box_class_scores = np.max(box_scores, axis=-1)
            filtering_mask = box_class_scores >= self.class_t

            filtered_boxes += boxes[i][filtering_mask].tolist()
            filtered_scores += box_class_scores[filtering_mask].tolist()
            filtered_classes += box_classes[filtering_mask].tolist()

        filtered_boxes = np.array(filtered_boxes)
        filtered_scores = np.array(filtered_scores)
        filtered_classes = np.array(filtered_classes)

        return filtered_boxes, filtered_classes, filtered_scores

    def non_max_suppression(
            self,
            filtered_boxes,
            box_classes,
            box_scores,
            iou_threshold=0.5):
        """ Performs Non-Maximum Suppression on filtered bounding boxes """

        box_predictions = list()
        predicted_box_classes = list()
        predicted_box_scores = list()

        for c in np.unique(box_classes):
            c_mask = box_classes == c
            c_boxes = filtered_boxes[c_mask]
            c_box_scores = box_scores[c_mask]

            while len(c_boxes) > 0:
                max_index = np.argmax(c_box_scores)
                max_box = c_boxes[max_index]
                max_score = c_box_scores[max_index]

                box_predictions.append(max_box)
                predicted_box_classes.append(c)
                predicted_box_scores.append(max_score)

                c_boxes = np.delete(c_boxes, max_index, axis=0)
                c_box_scores = np.delete(c_box_scores, max_index, axis=0)

                if len(c_boxes) == 0:
                    break

                ious = self.iou(max_box, c_boxes)
                c_boxes = c_boxes[ious < iou_threshold]
                c_box_scores = c_box_scores[ious < iou_threshold]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    def iou(self, box1, boxes):
        """
        Calculates Intersection over Union (IoU) between a box
        and an array of boxes
        """
        x1, y1, x2, y2 = box1
        boxes_x1, boxes_y1, boxes_x2, boxes_y2 = \
            boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        inter_x1 = np.maximum(x1, boxes_x1)
        inter_y1 = np.maximum(y1, boxes_y1)
        inter_x2 = np.minimum(x2, boxes_x2)
        inter_y2 = np.minimum(y2, boxes_y2)

        inter_area = np.maximum(inter_x2 - inter_x1, 0) * \
            np.maximum(inter_y2 - inter_y1, 0)
        box1_area = (x2 - x1) * (y2 - y1)
        boxes_area = (boxes_x2 - boxes_x1) * (boxes_y2 - boxes_y1)
        union_area = box1_area + boxes_area - inter_area

        ious = inter_area / union_area
        return ious
