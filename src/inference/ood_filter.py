"""
Production Out-of-Distribution and Unknown Sign Filter
Combines geometric feature space Mahalanobis/Euclidean centroid distance with confidence thresholds.
"""
import numpy as np
from typing import Tuple

class OODRejectionFilter:
    def __init__(self, train_centroid: np.ndarray, distance_threshold: float = 5.0, min_confidence: float = 0.18):
        self.train_centroid = train_centroid
        self.distance_threshold = distance_threshold
        self.min_confidence = min_confidence

    def filter(self, feature_vector: np.ndarray, probabilities: np.ndarray, classes: list) -> Tuple[str, float, bool]:
        if feature_vector is None or len(feature_vector) == 0 or np.all(feature_vector == 0):
            return "No Hand Detected", 0.0, False
            
        # 1. Geometric distance to training manifold
        dist = float(np.linalg.norm(feature_vector - self.train_centroid))
        if dist > self.distance_threshold:
            return "Unknown Sign / Out of Domain", 0.0, False
            
        # 2. Prediction confidence check
        max_idx = int(np.argmax(probabilities))
        max_conf = float(probabilities[max_idx])
        if max_conf < self.min_confidence:
            return "Uncertain", max_conf, False
            
        return classes[max_idx], max_conf, True
