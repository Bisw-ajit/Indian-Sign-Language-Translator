"""
Abstract Base Class for Interchangeable Hand Detectors
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np

class HandDetectionResult:
    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float, landmarks: Optional[np.ndarray] = None):
        """
        bbox: (x_min, y_min, x_max, y_max) in pixel coordinates
        confidence: detection confidence in [0.0, 1.0]
        landmarks: optional (21, 3) or (21, 2) normalized landmarks
        """
        self.bbox = bbox
        self.confidence = float(confidence)
        self.landmarks = landmarks

class BaseHandDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[HandDetectionResult]:
        """
        Detect hands in the input BGR image frame.
        Returns a list of HandDetectionResult objects.
        """
        pass
