# Scientific Data Leakage Audit Report: Indian Sign Language Translator

**Audit Date**: October 2026  
**Auditor**: Senior Machine Learning & Computer Vision Research Engineer  
**Status**: COMPLETE — LEAKAGE AUDITED & MITIGATED  

---

## 1. Executive Summary & Root Cause Analysis

A thorough inspection of the repository codebase (`Dataset_Synthesis/dataset_synthesis.py`, `Dataset_Synthesis/dataset_cleaning.py`, `Dataset_Preprocessing/dataset_preprocessing.py`, and `Model_Training/ModelTraining_Squeezenet.ipynb`) and academic thesis documentation (`Thesis/FYReport.pdf`) was conducted to uncover the mechanism behind the dramatic performance gap between reported thesis validation accuracy (~98–99%) and true out-of-distribution real-world benchmark performance (2.31%).

### Primary Audit Findings:
1. **Pre-Split Synthetic Augmentation Multiplier**:
   In `Dataset_Synthesis/dataset_synthesis.py`, each parent image is expanded into **18 synthetic child variants** (3 darkened $\times$ 2 blurred each = 6, 3 brightened $\times$ 2 blurred each = 6, plus 6 base lighting variants = 18 total).
   
2. **File Naming & Parent Linkage**:
   Child images are named with a simple suffix: `<original_filename>_<index>.jpg` (e.g., `image1_1.jpg`, `image1_2.jpg`, ...). The parent ID is embedded in the prefix before the final underscore.
   
3. **Un-Grouped Random Partitioning**:
   In `Model_Training/ModelTraining_Squeezenet.ipynb` (lines 252–253) and ImageAI's dataset intake, the split was performed by directory partitioning without parent-hash or subject grouping. Consequently:
   - If `image1_1.jpg`, `image1_3.jpg`, and `image1_7.jpg` are randomly placed into the `train` set, `image1_2.jpg` and `image1_4.jpg` land in the `val` set.
   - The validation set did NOT measure generalisation to novel gestures or distinct physical captures; rather, it evaluated nearest-neighbor memorization of virtually identical camera frames with minor illumination or blur variations.
   
4. **Signer & Session Metadata Absence**:
   The primary raw image archive was recorded without explicit subject/signer IDs in file metadata. All images originate from a single adult subject recorded under uniform black-background studio conditions.

---

## 2. Leakage Mechanics Diagram

```
[Original Raw Image: image_001.jpg]
                ↓
    [18x Synthetic Expansion]
   (image_001_1.jpg ... image_001_18.jpg)
                ↓
    [Standard Random 80/20 Split]
       ↙                     ↘
[Train Set (80%)]      [Val Set (20%)]
image_001_1.jpg        image_001_4.jpg  <-- SEVERE DATA LEAKAGE!
image_001_2.jpg        image_001_9.jpg      Exact same hand shape,
image_001_3.jpg                             same background, same signer.
```

---

## 3. Strict Remediation Protocol: Leakage-Safe Splitting

To ensure complete statistical validity and zero information leakage:

1. **Hierarchy of Splitting**:
   - **Step 1**: Identify base/parent physical captures. Strip any synthetic augmentation index suffixes (`_1`, `_2`, etc.) to resolve the canonical parent capture ID.
   - **Step 2**: If signer IDs exist, apply `GroupKFold` / group-aware splitting by `signer_id`. If signer IDs are uniform/absent, apply **Grouped Partitioning by Parent Capture ID**.
   - **Step 3**: Split into strictly isolated **TRAIN (70%)**, **VALIDATION (15%)**, and **LOCKED TEST (15%)** sets at the parent level.
   - **Step 4**: Apply augmentation and synthesis **ONLY** to the training split. Never augment or modify validation or test splits.

2. **Automated Verification**:
   The verification script `scripts/verify_no_leakage.py` implements:
   - Exact binary SHA-256 duplicate detection.
   - Perceptual difference / Mean-Squared-Error image similarity checks.
   - Parent ID prefix overlap checks across `(train ∩ val)`, `(train ∩ test)`, and `(val ∩ test)`.
   - Hard exit (`sys.exit(1)`) on any detected partition contamination.
