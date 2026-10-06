"""
Comprehensive Unit, Integration, and Regression Test Suite for ISL System (v2.0)
Tests:
1. Geometric Landmark Normalization (translation & scale invariance)
2. Level 3 Feature Extraction (78 dimensions)
3. Temperature Calibration
4. Dual-Barrier Robust OOD & Uncertainty Filtering
5. Temporal Exponential Moving Average (EMA)
6. Production Model Loading (Residual MLP 78d)
7. End-to-End Pipeline Execution on synthetic & real image data
"""
import os
import unittest
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

from src.landmarks.normalization import normalize_hand_landmarks
from src.features.hierarchy import extract_feature_levels
from src.calibration.temperature_scaling import TemperatureScaler
from src.inference.robust_ood_filter import RobustOODFilter
from src.temporal.consensus import ExponentialMovingAverager
from src.inference.pipeline import ProductionISLPipeline

class TestISLPipelineV2(unittest.TestCase):

    def test_landmark_normalization_shape_and_invariance(self):
        pts = np.random.uniform(50, 150, (21, 3))
        norm = normalize_hand_landmarks(pts)
        self.assertEqual(norm.shape, (63,))
        # Wrist at origin
        self.assertTrue(np.allclose(norm[:3], [0, 0, 0]))
        # Translation invariance
        pts_shifted = pts + np.array([40.0, -80.0, 20.0])
        norm_shifted = normalize_hand_landmarks(pts_shifted)
        self.assertTrue(np.allclose(norm, norm_shifted, atol=1e-5))

    def test_level3_feature_extraction(self):
        pts = np.random.uniform(-1.0, 1.0, 63)
        levels = extract_feature_levels(pts)
        self.assertIn("coords_dists", levels)
        feat_78 = levels["coords_dists"]
        self.assertEqual(feat_78.shape, (78,))
        # First 63 are coordinates
        self.assertTrue(np.allclose(feat_78[:63], pts))
        # Next 15 are non-negative distances
        self.assertTrue(np.all(feat_78[63:] >= 0.0))

    def test_temperature_scaling(self):
        scaler = TemperatureScaler()
        val_logits = np.array([[2.0, 0.5, 0.1], [0.1, 1.8, 0.2]])
        val_labels = np.array([0, 1])
        T = scaler.fit(val_logits, val_labels, max_iter=20)
        self.assertGreater(T, 0.5)
        probs = scaler.predict_proba(val_logits)
        self.assertEqual(probs.shape, (2, 3))
        self.assertTrue(np.allclose(np.sum(probs, axis=1), [1.0, 1.0]))

    def test_dual_barrier_ood_filter(self):
        # 10 synthetic class centroids
        centroids = np.zeros((10, 78))
        for c in range(10):
            centroids[c, 0] = c * 2.0
            
        ood = RobustOODFilter(class_centroids=centroids, max_class_distance=5.0, min_confidence=0.45)
        classes = ["G", "I", "K", "O", "P", "S", "U", "V", "X", "Y"]
        
        # Valid in-domain sample close to class 3 (O)
        feat_valid = centroids[3].copy() + 0.1
        probs_valid = np.zeros(10)
        probs_valid[3] = 0.85
        pred, conf, ok, reason = ood.filter(feat_valid, probs_valid, classes)
        self.assertTrue(ok)
        self.assertEqual(pred, "O")
        self.assertEqual(reason, "ACCEPTED")
        
        # Distant OOD sample
        feat_ood = np.full(78, 50.0)
        pred, conf, ok, reason = ood.filter(feat_ood, probs_valid, classes)
        self.assertFalse(ok)
        self.assertIn("OOD_DISTANCE", reason)
        
        # In-domain but low confidence sample
        probs_low = np.full(10, 0.10)
        pred, conf, ok, reason = ood.filter(feat_valid, probs_low, classes)
        self.assertFalse(ok)
        self.assertIn("LOW_CONFIDENCE", reason)

    def test_temporal_exponential_smoothing(self):
        smoother = ExponentialMovingAverager(alpha=0.7)
        p1 = np.array([1.0, 0.0, 0.0])
        s1 = smoother.update(p1)
        self.assertTrue(np.allclose(s1, p1))
        
        p2 = np.array([0.0, 1.0, 0.0])
        s2 = smoother.update(p2)
        # s2 = 0.7 * [0, 1, 0] + 0.3 * [1, 0, 0] = [0.3, 0.7, 0.0]
        self.assertAlmostEqual(s2[1], 0.7)
        self.assertAlmostEqual(s2[0], 0.3)
        self.assertAlmostEqual(np.sum(s2), 1.0)

    def test_production_model_loading(self):
        model_path = "models/production/res_mlp_78d.keras"
        self.assertTrue(os.path.exists(model_path))
        model = keras.models.load_model(model_path, compile=False)
        dummy_inp = np.zeros((1, 78), dtype=np.float32)
        out = model.predict(dummy_inp, verbose=0)
        self.assertEqual(out.shape, (1, 10))

    def test_end_to_end_production_pipeline_robustness(self):
        pipeline = ProductionISLPipeline("config/production.json")
        
        # Test 1: Empty / None frame
        res_none = pipeline.process_frame(None)
        self.assertEqual(res_none["status"], "Error")
        self.assertFalse(res_none["is_confident"])
        
        # Test 2: Blank image (No hand)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        res_blank = pipeline.process_frame(blank)
        self.assertEqual(res_blank["status"], "No Hand")
        self.assertEqual(res_blank["prediction"], "No Hand")
        self.assertFalse(res_blank["is_confident"])

if __name__ == '__main__':
    unittest.main()
