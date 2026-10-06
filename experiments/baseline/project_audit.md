# Scientific Project Audit: Indian Sign Language (ISL) Translator

**Audit Date**: October 4, 2026  
**Auditor**: Senior Machine Learning & Computer Vision Evaluation Engineer  
**Repository**: `Indian-Sign-Language-Translator`  
**Evaluation Standard**: Empirical, Reproducible, Zero-Fabrication Benchmark Protocol  

---

## 1. Project Architecture

The system is designed as a cascading multi-stage vision pipeline for translating static Indian Sign Language (ISL) gestures into text:

```
[Webcam Stream / Video Feed]
          │
          ▼
Stage 0: Face Detection & Activation (Haar Cascade, 35 consecutive frames)
          │
          ▼
Stage 1: Video Capture (7-second clip recorded at 24 FPS)
          │
          ▼
Stage 2: Frame Extraction & Sampling (Uniform random 20% sample selection)
          │
          ▼
Stage 3: Preprocessing Pipeline
   ├─► YOLOv3 Hand Detection & Localization (Darknet cfg + weights)
   ├─► Dynamic Bounding-Box Margin Expansion (Δx = ±54px, Δy = ±30px) & Cropping
   ├─► Spatial Standardization (Resize to 224 × 224 × 3 via PIL/OpenCV)
   └─► Dual-Space Skin Segmentation (Fused HSV [0,40,0]-[25,255,255] & YCbCr [0,138,67]-[255,173,133] + Watershed)
          │
          ▼
Stage 4: Deep Learning Classification & Voting Filter
   ├─► Fine-Tuned SqueezeNet CNN (Input: 224 × 224 × 3, Softmax: 10 Classes)
   └─► Consensus Decision (Majority Voting / Roulette Selection across sampled frames)
          │
          ▼
[Final Output ISL Character]
```

---

## 2. Dataset Structure & Audit

### 2.1 Datasets Described in Repository Artifacts
Based on the academic thesis (`Thesis/FYReport.pdf`), training notebook (`Model_Training/ModelTraining_Squeezenet.ipynb`), and presentation slides (`Thesis/FYPPT.pptx`):

1. **Primary Dataset (`ISL20C1200I` / `ISL25C1200I`)**:
   - **Original Classes**: 20–26 alphabet classes.
   - **Images per class**: 1,200 images per class (Total: 24,000 to 31,200 images).
   - **Split Ratio**: 80% Train, 20% Validation (documented in Section 4.5 of `FYReport.pdf`).
   - **Storage Location**: External archive hosted on Google Drive during training (`ds_rar_loc`). **Not checked into the GitHub repository** due to git LFS and size limits.
2. **Manually Created Stress-Test Set**:
   - **Class**: Single class ('O').
   - **Images**: 600 frames across 6 perturbation conditions:
     - Base condition (clear, standard lighting)
     - Motion blurred
     - Background interference (cluttered objects)
     - Distractor body parts
     - Skin-coloured background
     - Zoomed-out / scale shift
3. **Synthesized Dataset (`Dataset_Synthesis/`)**:
   - 18 synthetic variations generated per original image using brightness scaling ($0.4\times - 2.0\times$) and box blur ($r \in [5, 20]$).
   - Cleaned using YOLO confidence gating ($0.5$, $0.75$, $0.9$).

### 2.2 Classes Currently Deployed in Model
Extracted directly from `App/model_class.json`:
- **Total Classes**: 10
- **Alphabet Set**: `['G', 'I', 'K', 'O', 'P', 'S', 'U', 'V', 'X', 'Y']`
- **Class Mapping**:
  - `0`: G
  - `1`: I
  - `2`: K
  - `3`: O
  - `4`: P
  - `5`: S
  - `6`: U
  - `7`: V
  - `8`: X
  - `9`: Y

---

## 3. Model Architecture & Specifications

### 3.1 Classifier: SqueezeNet
- **Weights File**: `App/final_model.h5` (3,128,208 bytes / ~3.0 MB)
- **Framework**: Keras 2 / TensorFlow (saved in HDF5 format)
- **Input Dimensions**: `(None, 224, 224, 3)`
- **Output Dimensions**: `(None, 10)` with Softmax activation
- **Backbone**: Fire modules comprising $1 \times 1$ squeeze convolutions followed by expanded $1 \times 1$ and $3 \times 3$ filters.
- **Top Layer**: Global Average Pooling followed by dense classification layer.

### 3.2 Hand Detector: Darknet YOLOv3
- **Architecture File**: `App/yolo_models/cross-hands.cfg` (9,117 bytes)
- **Weights File**: `App/yolo_models/cross-hands.weights` (32,093,356 bytes / ~32 MB)
- **Input Resolution**: Standard $416 \times 416 \times 3$
- **Target Objects**: Class 0 = `"hand"`
- **Anchor Boxes & Scales**: 3-scale detection with Darknet-53 feature extractor.

### 3.3 Face Detector: Haar Cascade
- **Cascade Classifier**: OpenCV built-in `haarcascade_frontalface_default.xml`
- **Threshold**: 35 stable consecutive detection frames for activation.

---

## 4. Existing Evaluation Capabilities & Gaps

### 4.1 Existing Capabilities
- Inference engine on static `.jpg` images via `App/main.py:predictImage()`.
- Interactive prediction via Flask Web Studio (`web_app.py`).
- Automated end-to-end integration script (`test_end_to_end.py`).
- Benchmark timings recorded in `Thesis/FYReport.pdf` (Table 4.5.1 and Table 4.6.1).

### 4.2 Missing Capabilities Prior to This Audit
- **No standalone test harness**: The repository did not include automated scripts calculating precision, recall, F1, confusion matrices, or calibration errors.
- **No offline benchmark suite**: Evaluation was historically done ad-hoc within Google Colab notebooks or during live video capture.
- **Missing Ground-Truth Annotations for Hand Bounding Boxes**: No PASCAL VOC or COCO bounding box annotations are stored in the repo.
- **Missing Ground-Truth Segmentation Masks**: Hand masks were generated dynamically via watershed; no pixel-level ground-truth annotations exist.

---

## 5. Potential Sources of Data Leakage & Evaluation Bias

1. **Augmentation Leakage**:
   - In `Dataset_Synthesis/dataset_synthesis.py`, 18 variations are generated from each original image.
   - If an image dataset is split *after* synthesis without grouping by parent image hash/ID, synthesized variants of training images will leak into validation and test sets, artificially inflating reported validation accuracy (e.g., the reported ~98–99% validation accuracy).
2. **Subject / Signer Overfitting**:
   - The primary dataset `ISL20C1200I` was collected from a very limited number of human signers wearing dark clothing against a black background.
   - The network can memorize clothing texture, hand morphology, or skin tone rather than true semantic hand gestures.
3. **Temporal Redundancy**:
   - In video frame extraction, adjacent frames within a 7-second static hold clip are nearly identical. Random sampling across an entire un-partitioned clip results in near-duplicate frames between train and test.
4. **Clean Baseline vs Real-World Domain Shift**:
   - Original training images feature clean, dark, uniform studio backgrounds. Real-world camera feeds feature complex domestic/office backgrounds, variable illumination, and motion blur.

---

## 6. Audit Summary

| Component | Status | Verified Characteristics |
| :--- | :--- | :--- |
| **Model Weights (`final_model.h5`)** | Available | Valid HDF5 Keras model, 10 classes, input `(224, 224, 3)` |
| **YOLO Weights (`cross-hands.weights`)**| Available | Pre-trained Darknet YOLOv3 hand detector, ~32 MB |
| **Haar Cascades** | Available | Built-in OpenCV frontal face cascade |
| **Class Mapping (`model_class.json`)** | Available | 10 ISL characters (G, I, K, O, P, S, U, V, X, Y) |
| **Raw Training Dataset on Disk** | Missing | Excluded from git repository (~GBs); Colab RAR archive |
| **Historical Benchmark Records** | Available | Documented in `Thesis/FYReport.pdf` (48 pages) & `FYPPT.pptx` |
| **Automated Metric Suite** | Built in this Audit | `scikit-learn` & `matplotlib` test protocols |
