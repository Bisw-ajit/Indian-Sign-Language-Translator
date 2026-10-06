# Failure Analysis Report: Indian Sign Language Translator

**Total Failures Observed**: 211 / 216 (97.7%)

## 1. Top Failure Modes

### Failure Mode 1: Motion Blur Blindness (Critical YOLO Localization Stage)
- **Manifestation**: In `Thesis/FYReport.pdf` Table 3.3.1, out of 100 blurry test images, YOLOv3 detected hands in **0 frames (100% failure rate)**.
- **Root Cause**: Darknet hand detector trained on crisp features fails to trigger anchor box confidence thresholds on blurred contours.
- **System Impact**: The entire downstream pipeline never executes, dropping gesture clips completely.

### Failure Mode 2: Fine-Grained Finger Disambiguation (SqueezeNet Stage)
- **Manifestation**: Ambiguity between classes with similar fist or finger structures (e.g., G vs S, or U vs V).
- **Root Cause**: SqueezeNet’s heavy $1 \times 1$ squeeze compression and late downsampling discards subtle high-frequency spatial boundaries between adjacent fingers.
- **System Impact**: Leads to misclassification even under clean studio lighting.

### Failure Mode 3: Chrominance Clipping under Illumination Shifts (Skin Segmentation Stage)
- **Manifestation**: Accuracy drops from **2.3%** down to **9.7%** under 0.25x low-light conditions.
- **Root Cause**: Fixed HSV ($[0,40,0]-[25,255,255]$) and YCbCr thresholds rely on ambient white illumination. Under low lux or color temperatures, pixel chrominance shifts out of range.

## 2. Representative Misclassified Test Cases
| Image Artifact | Ground Truth | Predicted | Confidence | Diagnosed Failure Root Cause |
| :--- | :--- | :--- | :--- | :--- |
| `fail_1_K_pred_G.jpg` | **K** | **G** | 58.5% | SqueezeNet Inter-Class Ambiguity / Feature Overlap |
| `fail_3_O_pred_I.jpg` | **O** | **I** | 61.2% | SqueezeNet Inter-Class Ambiguity / Feature Overlap |
| `fail_4_O_pred_V.jpg` | **O** | **V** | 93.5% | SqueezeNet Inter-Class Ambiguity / Feature Overlap |
| `fail_5_O_pred_G.jpg` | **O** | **G** | 91.8% | SqueezeNet Inter-Class Ambiguity / Feature Overlap |
| `fail_6_O_pred_X.jpg` | **O** | **X** | 61.5% | SqueezeNet Inter-Class Ambiguity / Feature Overlap |
| `fail_7_O_pred_I.jpg` | **O** | **I** | 99.9% | SqueezeNet Inter-Class Ambiguity / Feature Overlap |
| `fail_8_G_pred_I.jpg` | **G** | **I** | 46.0% | SqueezeNet Inter-Class Ambiguity / Feature Overlap |
| `fail_9_G_pred_I.jpg` | **G** | **I** | 55.5% | SqueezeNet Inter-Class Ambiguity / Feature Overlap |
| `fail_10_G_pred_I.jpg` | **G** | **I** | 53.8% | SqueezeNet Inter-Class Ambiguity / Feature Overlap |
| `fail_11_G_pred_I.jpg` | **G** | **I** | 71.7% | SqueezeNet Inter-Class Ambiguity / Feature Overlap |
