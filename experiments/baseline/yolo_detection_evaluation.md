# YOLOv3 Hand Detection Quantitative Evaluation

## 1. Ground-Truth Bounding Box Status
- **Status**: Quantitative mAP@50 / IoU is **NOT AVAILABLE**.
- **Reason**: The repository does not contain PASCAL VOC, YOLO, or COCO format ground-truth bounding box annotation XML/JSON files for the dataset images.

## 2. Empirical Detection Success Rates (Measured & Thesis Table 3.3.1)
| Image Condition | Sample Count | YOLOv3 Detection Rate | Average Confidence | Detection Status |
| :--- | :--- | :--- | :--- | :--- |
| **Base Condition** | 89 | 100.0% (89/89) | 1.00 | Robust |
| **Skin-Coloured Background** | 100 | 100.0% (100/100) | 0.96 | Robust |
| **Background Interference** | 100 | 100.0% (100/100) | 0.97 | Robust |
| **Zoomed Out (Scale Shift)** | 100 | 100.0% (100/100) | 0.99 | Robust |
| **Face / Body Presence** | 100 | 77.0% (77/100) | 0.85 | Moderate |
| **Motion Blurred Hands** | 100 | **0.0% (0/100)** | N/A | **Critical Failure Mode** |

## 3. Average Detection Latency
- Measured on Apple Silicon CPU: **~640 ms - 670 ms per frame**
