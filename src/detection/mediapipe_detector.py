"""
MediaPipe Hand Detector and Landmarker Implementation
Uses Apple Metal GPU delegate for high performance and low latency on macOS arm64.
"""
import os
import cv2
import numpy as np
from typing import List, Optional
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.detection.base import BaseHandDetector, HandDetectionResult

class MediaPipeHandDetector(BaseHandDetector):
    def __init__(self, model_path: str = "src/models/hand_landmarker.task", min_detection_confidence: float = 0.5):
        self.min_detection_confidence = min_detection_confidence
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe hand landmarker model not found at {model_path}")
            
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.GPU
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> List[HandDetectionResult]:
        if frame is None or frame.size == 0:
            return []
            
        h, w = frame.shape[:2]
        # MediaPipe GPU on macOS requires SRGBA format
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba)
        
        detection_result = self.detector.detect(mp_img)
        results = []
        
        if not detection_result.hand_landmarks:
            return results
            
        for hand_idx, landmarks in enumerate(detection_result.hand_landmarks):
            # Extract 21 landmark coords in pixels
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            
            x_min = max(0, int(min(xs)))
            y_min = max(0, int(min(ys)))
            x_max = min(w, int(max(xs)))
            y_max = min(h, int(max(ys)))
            
            # Confidence score
            score = 0.9 # default high confidence for detected hand
            if detection_result.handedness and len(detection_result.handedness) > hand_idx:
                score = float(detection_result.handedness[hand_idx][0].score)
                
            # Convert landmarks to (21, 3) numpy array [x, y, z]
            lm_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
            
            results.append(HandDetectionResult(
                bbox=(x_min, y_min, x_max, y_max),
                confidence=score,
                landmarks=lm_array
            ))
            
        return results
