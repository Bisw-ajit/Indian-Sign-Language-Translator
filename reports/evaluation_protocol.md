# Evaluation Protocol: Indian Sign Language (ISL) Translator

**Protocol Version**: 1.0  
**Effective Date**: October 4, 2026  
**Auditor**: Senior ML & Computer Vision Evaluation Engineer  
**Reproducibility Seed**: `42`  

---

## 1. Objectives & Scope
The objective of this protocol is to establish a rigorous, reproducible, and scientifically grounded evaluation of the pre-trained SqueezeNet classifier and YOLOv3 hand detection pipeline deployed in the `Indian-Sign-Language-Translator` repository.

Evaluation rigorously distinguishes:
- **[MEASURED RESULT]**: Direct measurement computed on actual model executions, real extracted test samples, or controlled test perturbations.
- **[EXISTING REPOSITORY RESULT]**: Data, tables, and metrics officially reported in the repository's academic thesis (`Thesis/FYReport.pdf`) or presentation (`Thesis/FYPPT.pptx`).
- **[NOT AVAILABLE]**: Metrics that cannot be computed because prerequisite ground truth (e.g., COCO-style bounding box labels, pixel-level masks, or subject identities) is omitted from the repository.

---

## 2. Target Classes & Supported Vocabulary
The evaluated model classifies **10 ISL gestures**:

| Class Index | Sign Character | Hand Modality | Primary Pose Description |
| :---: | :---: | :---: | :--- |
| `0` | **G** | Two-handed | Closed fists stacked vertically |
| `1` | **I** | Single-handed | Extended upright little finger |
| `2` | **K** | Two-handed | Crossed index fingers with thumb contact |
| `3` | **O** | Single-handed | Curved thumb and fingers forming an 'O' ring |
| `4` | **P** | Two-handed | Downward index contact |
| `5` | **S** | Two-handed | Clasping fists / wrist lock |
| `6` | **U** | Single-handed | Index and middle fingers extended parallel |
| `7` | **V** | Single-handed | 'V' peace shape with index and middle |
| `8` | **X** | Two-handed | Crossed index fingers |
| `9` | **Y** | Single-handed | Extended thumb and pinky |

---

## 3. Test Set Construction & Leakage Prevention Strategy

### 3.1 Leakage Prevention
1. **Separation from Training Artifacts**:
   - The test set isolates real captured gesture frames and real test images from `Thesis/FYReport.pdf` (Figures 3.1.2, 4.4.2, 4.4.3, 6.2, 6.3, 6.4, 6.5).
   - Augmented images synthesized in `Dataset_Synthesis/modded_dataset` are excluded from the clean test evaluation to prevent post-augmentation leakage.
2. **Fixed Random Seed**:
   - All sampling, noise generation, and perturbation experiments utilize `numpy.random.seed(42)` and `random.seed(42)` for deterministic reproduction.
3. **Controlled Multi-Condition Perturbation Suite**:
   - Perturbations (illumination, background clutter, Gaussian noise, box blur, JPEG compression, rotation, translation, scaling) are generated dynamically on evaluation copies and strictly never stored in or fed back into training pipelines.

---

## 4. Evaluation Metrics Formulation

### 4.1 Classification Metrics
For a 10-class problem with classes $C = \{0, \dots, 9\}$ and confusion matrix $M$ where $M_{ij}$ is the count of true class $i$ predicted as class $j$:
- **Overall Accuracy**: $\frac{\sum_i M_{ii}}{\sum_{i,j} M_{ij}}$
- **Balanced Accuracy**: $\frac{1}{|C|} \sum_i \frac{M_{ii}}{\sum_j M_{ij}}$
- **Macro Precision**: $\frac{1}{|C|} \sum_j \frac{M_{jj}}{\sum_i M_{ij}}$
- **Macro Recall**: $\frac{1}{|C|} \sum_i \frac{M_{ii}}{\sum_j M_{ij}}$
- **Macro F1 Score**: $\frac{1}{|C|} \sum_i \frac{2 \cdot P_i \cdot R_i}{P_i + R_i}$
- **Cohen's Kappa ($\kappa$)**: $\frac{p_o - p_e}{1 - p_e}$
- **Matthews Correlation Coefficient (MCC)**: Multi-class formulation measuring prediction correlation.
- **Expected Calibration Error (ECE)**: $ECE = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$ across 10 confidence bins.
- **Brier Score**: Multi-class quadratic loss $\frac{1}{N} \sum_{n=1}^N \sum_{k=1}^K (p_{nk} - y_{nk})^2$.

### 4.2 Temporal Consensus Metrics
- **Frame Accuracy**: Accuracy evaluated independently on isolated video frames.
- **Clip Accuracy**: Accuracy after majority voting across $K$ sampled frames of a 7-second gesture clip.
- **Voting Gain**: $\text{Voting Gain} = \text{Clip Accuracy} - \text{Frame Accuracy}$.

---

## 5. Artifact Hashes & Execution Environment
- **Model File**: `App/final_model.h5`
  - Size: 3,128,208 bytes
  - Architecture: SqueezeNet (Transfer Learning via ImageAI / Keras)
- **Detector Weights**: `App/yolo_models/cross-hands.weights`
  - Size: 32,093,356 bytes
- **Host Runtime**: Apple Silicon macOS, Python 3.10.20, TensorFlow 2.21.0, OpenCV 4.14.0.94, scikit-learn 1.7.2.
