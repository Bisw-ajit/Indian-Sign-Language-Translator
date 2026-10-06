# Scientific Classification Failure Analysis: Indian Sign Language Recognition Pipeline

**Audit Date**: October 2026  
**Auditor**: Lead ML & Computer Vision Research Team  
**Evaluation Target**: Current Production Model (`models/production/landmark_mlp.keras`)  
**Locked Test Suite**: 150 stratified test samples spanning 10 classes (`G, I, K, O, P, S, U, V, X, Y`) across unseen test signers (`signer_04`, `signer_05`).

---

## 1. Executive Summary & Forensic Diagnosis

The current system achieves **22.67% classification accuracy** and **0.1958 Macro F1** on the locked real-world held-out test suite. While this represents a massive gain over the original 2.31% baseline, it falls far short of a production-grade sign language translator.

A thorough empirical failure analysis was performed by inspecting:
- `reports/final_confusion_matrix.csv` & `reports/final_confusion_matrix.png`
- Per-class metrics recorded in `reports/class_metrics.csv`
- Error confusion pairs in `reports/top_confusion_pairs.csv`
- Confidence distribution of correct vs incorrect predictions (`reports/error_analysis_visualizations/confidence_distribution.png`)
- Sub-population breakdowns across signers, lighting, and background conditions.

---

## 2. Quantitative Per-Class Error Breakdown

From [reports/class_metrics.csv](file:///Users/biswajit/Downloads/CLG%20PROJECT/Indian-Sign-Language-Translator/reports/class_metrics.csv):

| Class | Support | TP | FP | FN | Precision | Recall | F1 Score | Diagnosis |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **G** | 15 | 0 | 0 | 15 | 0.0000 | 0.0000 | **0.0000** | 100% Collapse (12 mispredicted as O) |
| **I** | 15 | 0 | 1 | 15 | 0.0000 | 0.0000 | **0.0000** | 100% Collapse (13 mispredicted as O) |
| **K** | 15 | 9 | 2 | 6 | 0.8182 | 0.6000 | **0.6923** | **Strongest performing class** |
| **O** | 15 | 15 | 107 | 0 | 0.1230 | 1.0000 | **0.2190** | **Massive False Positive Sink (107 FP)** |
| **P** | 15 | 3 | 0 | 12 | 1.0000 | 0.2000 | **0.3333** | High precision, low recall (12 to O) |
| **S** | 15 | 0 | 0 | 15 | 0.0000 | 0.0000 | **0.0000** | 100% Collapse (15 mispredicted as O) |
| **U** | 15 | 1 | 5 | 14 | 0.1667 | 0.0667 | **0.0952** | Severe misprediction (14 to O) |
| **V** | 15 | 0 | 0 | 15 | 0.0000 | 0.0000 | **0.0000** | 100% Collapse (15 mispredicted as O) |
| **X** | 15 | 1 | 1 | 14 | 0.5000 | 0.0667 | **0.1176** | Severe misprediction (11 to O) |
| **Y** | 15 | 5 | 0 | 10 | 1.0000 | 0.3333 | **0.5000** | Distinct geometry (thumb+pinky spread) |

### Key Observations:
1. **Four classes completely collapsed (0.00% Recall)**: Classes `G`, `I`, `S`, and `V`.
2. **Two classes moderately survive**: Class `K` (60.0% recall, 0.69 F1) and Class `Y` (33.3% recall, 0.50 F1). Both exhibit distinctive geometric finger expansions that separate them in coordinate space.
3. **Class O is an overwhelming attractor / default sink**:
   - Out of 150 test samples, **122 samples (81.3%)** were predicted as Class `O`.
   - Class `O` has 100% recall (15/15) but abysmal precision (12.3%) due to 107 false positives.

---

## 3. Top Confusion Pairs & Forensic Mechanism

From [reports/top_confusion_pairs.csv](file:///Users/biswajit/Downloads/CLG%20PROJECT/Indian-Sign-Language-Translator/reports/top_confusion_pairs.csv):

1. **S → O**: 15 / 15 (100.0%)
2. **V → O**: 15 / 15 (100.0%)
3. **U → O**: 14 / 15 (93.3%)
4. **I → O**: 13 / 15 (86.7%)
5. **G → O**: 12 / 15 (80.0%)
6. **P → O**: 12 / 15 (80.0%)
7. **X → O**: 11 / 15 (73.3%)
8. **Y → O**: 9 / 15 (60.0%)
9. **K → O**: 6 / 15 (40.0%)

### Root Cause Analysis of the "Class O Sink":
During Phase 2 dataset generation, synthetic geometric fallback routines for contour-based landmark extraction were invoked when the detector missed stylized cartoon graphics.
- Specifically, circular and closed hand profiles generated invariant, degenerate contour points centered at `[0, 1]` with **zero intra-class variance** (`std = 0.0000`).
- Because Class `O` had degenerate variance, the MLP's softmax decision boundaries collapsed towards predicting Class `O` whenever a test sample exhibited ambiguous or low-magnitude relative coordinate displacements.

---

## 4. Confidence Distribution Analysis

From our empirical measurements:
- **Mean Confidence for Correct Predictions**: **57.06%**
- **Mean Confidence for Incorrect Predictions**: **16.34%**
- **Confidence Gap**: **+40.72 percentage points**

Unlike the original baseline SqueezeNet (which was frequently and confidently wrong at >58% confidence on false predictions), the calibrated production MLP exhibits healthy confidence separation: incorrect predictions collapse towards low confidence, enabling effective downstream thresholding.

---

## 5. Sub-Population Breakdown: Signer, Lighting & Background

### A. Signer Generalization
- **Signer 04** (Unseen): **27.50% Accuracy**
- **Signer 05** (Unseen): **17.14% Accuracy**
- **Disparity**: A **10.36 pp gap** exists between the two unseen test signers, indicating significant cross-subject morphological sensitivity in hand proportions.

### B. Lighting Conditions
- Shadows: **28.57%**
- Cool: **23.81%**
- Low Light: **23.81%**
- Warm: **22.73%**
- Bright: **22.73%**
- Normal: **19.05%**
- Backlit: **18.18%**
- **Finding**: Performance is remarkably stable across illuminations (range: 18.2% to 28.6%), confirming that landmark geometry successfully removed raw pixel color dependency.

### C. Background Conditions
- Cluttered: **33.33%**
- Visually Complex: **23.33%**
- Room: **23.33%**
- Plain: **20.00%**
- Outdoor: **13.33%**

---

## 6. Ranked List of Top 5 Experimentally Supported Problems

Based on our empirical analysis, the 5 core bottlenecks constraining accuracy are:

1. **Problem 1: Degenerate Representation Sink in Class O**:
   Class `O` acts as an attractor absorbing 81.3% of all model predictions because its training feature distribution lacks spatial variance.
2. **Problem 2: Single-Hand Limitation on Two-Handed ISL Gestures**:
   Indian Sign Language contains inherently two-handed alphabets (e.g., `G` is two stacked closed fists, `K` and `P` utilize interaction between both hands). The current feature extractor flattens only the primary hand into 63 coordinates, discarding inter-hand spatial interactions.
3. **Problem 3: Absence of Explicit Biomechanical Finger Geometry**:
   Raw normalized $(x, y, z)$ coordinates do not explicitly encode finger flexion angles, inter-finger distances (e.g., distance between index and middle fingertip in `U` vs `V`), or thumb opposition. Classifiers struggle to separate subtle geometric distinctions from raw coordinates alone.
4. **Problem 4: Stylized/Synthetic Artifact Sensitivity in Detector**:
   MediaPipe's machine-learning detector was trained on natural human hands and achieves 58.3% detection on natural captures, but struggles when presented with synthetic or simplified hand silhouettes.
5. **Problem 5: Linear / Flat Softmax Boundary Collapse**:
   The shallow MLP with a single bottleneck layer without feature scaling or angular margin loss collapses under high-dimensional coordinate ambiguity, requiring non-linear kernelized or tree-based ensemble separation.
