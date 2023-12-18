from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops, roi_align
import torchvision.models.detection._utils as det_utils

from utils import decode_boxes


class RoIHeads(nn.Module):
    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # Mask
    ):
        super().__init__()

        self.box_similarity = box_ops.box_iou

        bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def postprocess_detections(
        self,
        class_logits,
        box_regression,
        proposals,
        image_shape
    ):
        num_classes = class_logits.shape[-1]

        proposals = proposals.unsqueeze(1).repeat(1, num_classes, 1).view(-1, 4)
        box_regression = box_regression.view(-1, 4)
        boxes = decode_boxes(box_regression, proposals, weights=(10.0, 10.0, 5.0, 5.0))

        boxes = boxes.view(-1, num_classes, 4)
        scores = F.softmax(class_logits, -1)
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes)
        labels = labels.view(1, -1).expand_as(scores)

        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]

        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        inds = torch.where(scores > self.score_thresh)[0]

        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # non-maximum suppression, independently done per class
        keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
        # keep only topk scoring predictions
        keep = keep[: self.detections_per_img]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        return boxes, scores, labels

    def forward(
        self,
        features,
        proposals,
        image_shape,
    ):
        box_features = self.box_roi_pool(features, [proposals], [image_shape])
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shape)
        return boxes, scores, labels