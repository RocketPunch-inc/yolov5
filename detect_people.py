# encoding: utf-8
""" Similar to detect.py, but initialize model just once and keep using it.
"""
import numpy as np
import torch
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression
from utils.torch_utils import select_device


class DetectPeople():
    """ yolo v5 supports various options, but only a few will be used in here.

        We're going to use yolo5s.pt, the lowest score but most fast and small.
        That's ok because detecting person is all we need.
    """
    def __init__(self):
        device = 'cpu'
        weights = 'yolov5s.pt'

        self.device = select_device(device)
        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        self.imgsz = 640  # default
        self.conf_thres = 0.25  # default
        self.iou_thres = 0.45  # default
        self.agnostic_nms = False  # default

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        # We will only use class 0 (person), but make sure it's person.
        assert self.names[0] == 'person', 'Class 0 must be person, but it is {}'.format(self.names[0])

    def detect(self, img0):
        """ from utils/datasets:LoadImages
        """

        # Padded resize
        img = letterbox(img0, new_shape=self.imgsz)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=[0], agnostic=self.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                for *xyxy, conf, cls in det:
                    if conf > 0.5:  # value determined by heuristic
                        return True

        return False
