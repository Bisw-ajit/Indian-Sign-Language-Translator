# Hand Skin Segmentation Evaluation

## 1. Pixel-Level Ground-Truth Masks Status
- **Status**: Quantitative Mean IoU and Dice Coefficient are **NOT AVAILABLE**.
- **Reason**: The repository does not store manually annotated binary ground-truth segmentation masks; masks are generated dynamically via HSV + YCbCr color thresholding and watershed.

## 2. Qualitative & Algorithmic Analysis
- **HSV Space Masking**: `[0, 40, 0]` to `[25, 255, 255]`
- **YCbCr Space Masking**: `[0, 138, 67]` to `[255, 173, 133]`
- **Strengths**: Successfully strips black and neutral background pixels without deep learning overhead (~5 ms latency).
- **Failure Modes**:
  1. **Lighting Shifts**: Severe underexposure causes skin chrominance to fall outside the HSV/YCbCr thresholds, causing finger clipping.
  2. **Complex Skin-Tone Backgrounds**: Non-hand skin-tone objects pass the color mask and trigger watershed leaks.
