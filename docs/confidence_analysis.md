# Classification Confidence & Calibration Analysis

**Total Evaluated Samples**: 216  

## 1. Confidence Summary
- **Mean Confidence (Correct Predictions)**: 0.5848 (58.48%)
- **Mean Confidence (Incorrect Predictions)**: 0.5873 (58.73%)
- **Minimum Confidence**: 0.3000 (30.00%)
- **Maximum Confidence**: 1.0000 (100.00%)
- **Standard Deviation of Confidence**: 0.1886

## 2. Model Calibration
- **Expected Calibration Error (ECE)**: 0.5641 (56.41%)
- **Multi-class Brier Score**: 1.3552

## 3. Analysis & Overconfidence Findings
- The model exhibits a confidence gap of 0.0025 between correct and erroneous classifications.
- Incorrect predictions still have an average confidence of 58.73%, indicating the model can be **confidently wrong** when presented with ambiguous or degraded gestures.
