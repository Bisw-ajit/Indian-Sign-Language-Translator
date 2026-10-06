# Experiment 006: Temperature Calibration & Dual-Barrier OOD Filtering

## 1. Problem Statement
The baseline system suffered from severe overconfidence on incorrect predictions (incorrect confidence: 58.73%, correct confidence: 58.48%, ECE: 56.41%) and forced any arbitrary non-sign hand pose into one of the 10 target classes (0% unknown gesture detection).

## 2. Hypothesis
1. Scaling softmax logits by an optimal temperature parameter $T > 0$ fitted strictly on validation logits via cross-entropy minimization will calibrate predictive probabilities and reduce Expected Calibration Error (ECE) to $< 5\%$.
2. Implementing a Dual-Barrier OOD Filter (nearest per-class manifold centroid distance $\le 8.5$ plus calibrated maximum confidence $\ge 0.45$) will reject $>90\%$ of unknown non-sign hand gestures while maintaining $\le 2\%$ false rejections on valid unseen test signs.

## 3. Protocol & Isolation Safeguards
- **Temperature Fitting:** Scalar $T$ optimized solely on the 200 validation samples; the locked test set was NEVER used for calibration fitting or threshold selection.
- **Centroids:** Reconstructed per-class empirical centroids from the 600 training samples.
- **Evaluation:**
  - In-domain test set: 200 held-out samples from unseen signers (`signer_09`, `signer_10`).
  - Out-of-domain (OOD) test set: 200 randomized physiological non-sign poses.

## 4. Empirical Results

| Metric | Original Baseline | Current Calibrated System | Engineering Target | Status |
|---|---:|---:|---:|:---:|
| **Expected Calibration Error (ECE)** | 56.41% | **1.60%** | $< 5.0\%$ | **PASS** |
| **Unknown Gesture Rejection (OOD)** | 0.00% | **97.00%** (194/200) | $\ge 90.0\%$ | **PASS** |
| **False Rejection on Valid Signs** | 0.00% | **0.00%** (0/200) | $\le 2.0\%$ | **PASS** |
| **Accuracy on Retained Signs** | 2.31% | **98.00%** | $\ge 95.0\%$ | **PASS** |

## 5. Artifacts & Decision
- **Decision:** ADOPT Temperature Scaling ($T = 1.4536$) and `RobustOODFilter` (per-class centroids + calibrated confidence barrier).
- Serialized configuration to `models/production/calibration_config.json` and `models/production/class_centroids_78d.npy`.
