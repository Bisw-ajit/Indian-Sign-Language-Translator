"""
Unknown and Uncertain Rejection Filter
Detects when the model is presented with non-sign gestures, background noise, or low-confidence inputs.
"""
import numpy as np
from typing import Tuple, Optional

class RejectionFilter:
    def __init__(self, confidence_threshold: float = 0.35, entropy_threshold: float = 2.1):
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold

    def filter(self, probabilities: np.ndarray, classes: list) -> Tuple[str, float, bool]:
        """
        Returns (predicted_class_or_status, confidence, is_accepted)
        Possible statuses:
        - '<Class>' if accepted
        - 'Uncertain' if confidence is low
        - 'Unknown / No Gesture' if entropy is high
        """
        if probabilities is None or len(probabilities) == 0:
            return "No Hand", 0.0, False
            
        max_idx = int(np.argmax(probabilities))
        max_conf = float(probabilities[max_idx])
        
        # Calculate Shannon entropy
        eps = 1e-12
        entropy = -float(np.sum(probabilities * np.log(probabilities + eps)))
        
        if max_conf < self.confidence_threshold:
            return "Uncertain", max_conf, False
        if entropy > self.entropy_threshold:
            return "Unknown / Ambiguous", max_conf, False
            
        return classes[max_idx], max_conf, True
