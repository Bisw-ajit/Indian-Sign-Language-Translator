"""
Temporal Consensus and Smoothing Engine
Implements:
1. Sliding Window Probability Averaging
2. Exponential Moving Average (EMA)
3. Confidence-Weighted Window Voting
4. Temporal State Tracker with Hysteresis / Cooldown
"""
import numpy as np
from typing import List, Dict, Any, Optional

class SlidingWindowAverager:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history: List[np.ndarray] = []

    def reset(self):
        self.history.clear()

    def update(self, probs: np.ndarray) -> np.ndarray:
        self.history.append(probs)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        return np.mean(self.history, axis=0)


class ExponentialMovingAverager:
    def __init__(self, alpha: float = 0.4):
        """
        alpha: weight for new frame (0 < alpha <= 1).
        Smoothed: S_t = alpha * X_t + (1 - alpha) * S_{t-1}
        """
        self.alpha = alpha
        self.smoothed: Optional[np.ndarray] = None

    def reset(self):
        self.smoothed = None

    def update(self, probs: np.ndarray) -> np.ndarray:
        if self.smoothed is None:
            self.smoothed = probs.copy()
        else:
            self.smoothed = self.alpha * probs + (1.0 - self.alpha) * self.smoothed
        # Renormalize to sum to 1
        s = np.sum(self.smoothed)
        if s > 0:
            self.smoothed /= s
        return self.smoothed


class ConfidenceWeightedSmoother:
    def __init__(self, window_size: int = 5, power: float = 2.0):
        self.window_size = window_size
        self.power = power
        self.history: List[np.ndarray] = []

    def reset(self):
        self.history.clear()

    def update(self, probs: np.ndarray) -> np.ndarray:
        self.history.append(probs)
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
        # Weight each frame by max(prob)^power
        weights = [float(np.max(p) ** self.power) for p in self.history]
        total_w = sum(weights)
        if total_w == 0:
            return np.mean(self.history, axis=0)
            
        w_probs = sum(w * p for w, p in zip(weights, self.history)) / total_w
        return w_probs
