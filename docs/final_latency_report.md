# Final System Latency, Throughput & Profiling Report

**Date**: October 2026  
**Hardware Evaluated**: Apple Silicon M2 (arm64, 8 cores, Metal GPU), macOS Darwin  
**Measurement Protocol**: 50 repetitions with warm-up cycles, microsecond precision timer  

---

## 1. End-to-End Latency Comparison

| Stage | Baseline Pipeline | Production Pipeline | Latency Reduction |
| :--- | :---: | :---: | :---: |
| **Hand Detection** | 197.49 ms (Darknet YOLOv3) | **4.02 ms** (MediaPipe Metal) | **-97.96%** |
| **Feature Extraction / Preprocessing**| 34.40 ms (HSV+YCbCr+Watershed) | **0.00 ms** (Geometric Normalization) | **-100.00%** |
| **Classification & Filtering** | 10.31 ms (SqueezeNet) | **20.49 ms** (Landmark MLP + OOD Filter) | +98.7% |
| **Total Pipeline Latency (Mean)** | **242.20 ms** | **24.51 ms** | **-89.88%** |
| **P95 Latency** | **320.21 ms** | **24.34 ms** | **-92.40%** |
| **Throughput (FPS)** | **4.13 FPS** | **40.80 FPS** | **+887.89%** |

---

## 2. Memory Consumption & Storage

| Metric | Baseline Architecture | Production Pipeline | Delta |
| :--- | :---: | :---: | :---: |
| **Process RAM Footprint** | ~1,940 MB (1.94 GB) | **503.12 MB (~0.49 GB)** | **-74.07%** |
| **Model On-Disk Footprint** | 32.0 MB YOLO + 2.98 MB SqueezeNet | **7.8 MB Detector + 0.06 MB MLP** | **-77.65%** |
| **Parameters Count** | 62M (Darknet) + 727K (SqueezeNet) | **17,546 (MLP Classifier)** | **-99.97%** |
