"""
YOLOv3 Hand Detector Implementation
"""
import os
import cv2
import numpy as np
from typing import List
from src.detection.base import BaseHandDetector, HandDetectionResult

class YOLOv3HandDetector(BaseHandDetector):
    def __init__(self, cfg_path: str, weights_path: str, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        if not os.path.exists(cfg_path) or not os.path.exists(weights_path):
            raise FileNotFoundError(f"YOLOv3 model files not found: cfg={cfg_path}, weights={weights_path}")
            
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        layer_names = self.net.getLayerNames()
        out_layers_indices = self.net.getUnconnectedOutLayers()
        if len(out_layers_indices.shape) == 1:
            self.output_layers = [layer_names[i - 1] for i in out_layers_indices]
        else:
            self.output_layers = [layer_names[i[0] - 1] for i in out_layers_indices]

    def detect(self, frame: np.ndarray) -> List[HandDetectionResult]:
        if frame is None or frame.size == 0:
            return []
            
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                conf = float(scores[class_id])
                
                if conf >= self.confidence_threshold:
                    box = detection[0:4] * np.array([w, h, w, h])
                    centerX, centerY, width, height = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(conf))
                    
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, bw, bh = boxes[i]
                x_min = max(0, x)
                y_min = max(0, y)
                x_max = min(w, x + bw)
                y_max = min(h, y + bh)
                results.append(HandDetectionResult((x_min, y_min, x_max, y_max), confidences[i]))
                
        return results
