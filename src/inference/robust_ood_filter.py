"""
Production Out-of-Distribution (OOD) and Unknown Sign Filter
Employs a Dual-Verification Barrier:
1. Per-Class Centroid Metric: Measures minimum Euclidean distance to nearest sign manifold center.
2. Calibrated Maximum Softmax Probability (MSP): Rejects low-confidence/ambiguous distributions.
"""
import numpy as np
from typing import Tuple, List, Optional

class RobustOODFilter:
    def __init__(
        self,
        class_centroids: np.ndarray, # shape (10, D)
        max_class_distance: float = 7.5,
        min_confidence: float = 0.55
    ):
        self.class_centroids = class_centroids
        self.max_class_distance = max_class_distance
        self.min_confidence = min_confidence

    def filter(
        self,
        feature_vector: np.ndarray,
        probabilities: np.ndarray,
        classes: List[str]
    ) -> Tuple[str, float, bool, str]:
        """
        Returns:
            (predicted_class, confidence, is_valid, rejection_reason)
        """
        if feature_vector is None or len(feature_vector) == 0:
            return "No Hand", 0.0, False, "EMPTY_INPUT"

        # 1. Manifold proximity: minimum distance to any class centroid
        dists = np.linalg.norm(self.class_centroids - feature_vector, axis=1)
        min_dist = float(np.min(dists))
        nearest_class_idx = int(np.argmin(dists))
        
        if min_dist > self.max_class_distance:
            return "Unknown Sign", 0.0, False, f"OOD_DISTANCE_{min_dist:.2f}"

        # 2. Confidence threshold
        max_idx = int(np.argmax(probabilities))
        max_conf = float(probabilities[max_idx])
        
        if max_conf < self.min_confidence:
            return "Uncertain", max_conf, False, f"LOW_CONFIDENCE_{max_conf:.2f}"

        return classes[max_idx], max_conf, True, "ACCEPTED"
