"""
Production Pipeline Inference Engine (v2.0)
Integrates:
- BaseHandDetector (MediaPipe with GPU/CPU fallback)
- Geometric Normalization + Level 3 Feature Extraction (78 dims)
- Calibrated Residual MLP (Res-Dense) Classifier
- Dual-Barrier Robust OOD & Uncertainty Rejection Filter
- Exponential Moving Average (EMA, alpha=0.7) Temporal Consensus
- Graceful Error Handling (zero crashes on missing hand, occlusion, or corrupt frames)
"""
import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any, Optional

from src.detection.mediapipe_detector import MediaPipeHandDetector
from src.landmarks.normalization import normalize_hand_landmarks
from src.features.hierarchy import extract_feature_levels
from src.inference.robust_ood_filter import RobustOODFilter
from src.temporal.consensus import ExponentialMovingAverager

class ProductionISLPipeline:
    def __init__(self, config_path: str = "config/production.json"):
        self.classes = ["G", "I", "K", "O", "P", "S", "U", "V", "X", "Y"]
        self.detector = None
        self.classifier = None
        self.ood_filter = None
        self.smoother = None
        self.temperature = 1.4536
        self.config = {}
        
        # Check config path, fallback to configs/ if needed
        if not os.path.exists(config_path) and os.path.exists("configs/production.json"):
            config_path = "configs/production.json"
            
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
            
        self.load_components()

    def load_components(self):
        # 1. Detector
        det_path = "models/production/hand_landmarker.task"
        if not os.path.exists(det_path):
            det_path = "src/models/hand_landmarker.task"
        self.detector = MediaPipeHandDetector(model_path=det_path, min_detection_confidence=0.3)
        
        # 2. Classifier
        clf_path = self.config.get("classifier", {}).get("model_path", "models/production/res_mlp_78d.keras")
        if not os.path.exists(clf_path):
            clf_path = "experiments/exp_003_classifier_comparison/res_mlp_model.keras"
        self.classifier = keras.models.load_model(clf_path, compile=False)
        self.temperature = float(self.config.get("classifier", {}).get("temperature", 1.4536))
        
        # 3. Dual-Barrier OOD Filter
        centroids_path = self.config.get("ood_rejection", {}).get("centroids_path", "models/production/class_centroids_78d.npy")
        if os.path.exists(centroids_path):
            centroids = np.load(centroids_path)
        else:
            centroids = np.zeros((10, 78), dtype=np.float32)
            
        max_dist = float(self.config.get("ood_rejection", {}).get("max_class_distance", 8.5))
        min_conf = float(self.config.get("ood_rejection", {}).get("min_confidence", 0.45))
        self.ood_filter = RobustOODFilter(class_centroids=centroids, max_class_distance=max_dist, min_confidence=min_conf)
        
        # 4. Temporal Smoother (EMA alpha=0.7)
        alpha = float(self.config.get("temporal_smoothing", {}).get("ema_alpha", 0.70))
        self.smoother = ExponentialMovingAverager(alpha=alpha)

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Processes a single BGR camera frame with comprehensive error handling.
        Never crashes on invalid input, missing hand, or corrupt data.
        """
        if frame is None or frame.size == 0:
            return {
                "status": "Error",
                "message": "Invalid or empty frame",
                "prediction": "None",
                "confidence": 0.0,
                "is_confident": False,
                "bbox": None
            }

        try:
            # 1. Detection
            detections = self.detector.detect(frame)
            if not detections or detections[0].landmarks is None:
                self.smoother.reset()
                return {
                    "status": "No Hand",
                    "prediction": "No Hand",
                    "confidence": 0.0,
                    "is_confident": False,
                    "bbox": None
                }
                
            # Primary hand
            primary = detections[0]
            bbox = primary.bbox
            
            # 2. Geometric Normalization & Feature Extraction (Level 3: 78 dims)
            raw_norm_63 = normalize_hand_landmarks(primary.landmarks)
            feat_levels = extract_feature_levels(raw_norm_63)
            feat_78 = feat_levels["coords_dists"] # (78,)
            
            # 3. Model Inference (Softmax probabilities)
            raw_probs = self.classifier.predict(np.expand_dims(feat_78, 0), verbose=0)[0]
            
            # 4. Calibration (Logit Temperature Scaling)
            # Reconstruct logits approximation: log(p) + C, scale by T
            eps = 1e-9
            approx_logits = np.log(raw_probs + eps)
            scaled_logits = approx_logits / self.temperature
            exp_scaled = np.exp(scaled_logits - np.max(scaled_logits))
            calibrated_probs = exp_scaled / np.sum(exp_scaled)
            
            # 5. Temporal Smoothing (EMA)
            smoothed_probs = self.smoother.update(calibrated_probs)
            
            # 6. Dual-Barrier OOD & Uncertainty Filtering
            pred_class, conf, ok, reason = self.ood_filter.filter(feat_78, smoothed_probs, self.classes)
            
            return {
                "status": "Success" if ok else "Filtered",
                "prediction": pred_class,
                "confidence": float(conf),
                "is_confident": ok,
                "filter_reason": reason,
                "bbox": bbox,
                "landmarks": primary.landmarks
            }
            
        except Exception as e:
            return {
                "status": "Exception Handled",
                "message": str(e),
                "prediction": "None",
                "confidence": 0.0,
                "is_confident": False,
                "bbox": None
            }
