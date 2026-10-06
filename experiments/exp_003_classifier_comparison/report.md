# Experiment 003: Classifier Architecture Comparison

## 1. Problem Statement
With the Level 3 (78-dimensional) feature representation established (63 normalized 3D landmarks + 15 inter-fingertip and wrist Euclidean distances), we must establish which classifier architecture achieves the highest accuracy and macro F1 on unseen signers while remaining within production latency constraints ($\le 30$ ms per frame).

## 2. Hypothesis
A Residual MLP (Res-Dense) featuring skip connections across dense feature representations will mitigate representation degradation and gradient vanishing, yielding superior generalization on unseen signers compared to a vanilla fully connected MLP and 1D-CNN, while maintaining sub-20ms inference latency.

## 3. Experimental Protocol
- **Dataset:** 78-dimensional features (`real_world_dataset/{train,validation,test}_landmarks_78d.npz`).
- **Splits:**
  - Train: 6 signers (600 samples)
  - Validation: 2 signers (200 samples)
  - Locked Test: 2 signers (`signer_09`, `signer_10` — 200 samples)
- **Protocol:** Strict locked test evaluation, identical batch sizes (32), Adam optimizer, early stopping on validation loss.

## 4. Empirical Results

| Architecture | Test Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Latency (mean) | P95 Latency | FPS | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Landmark MLP (FC)** | 95.00% | 95.00% | 94.94% | 94.94% | 17.92 ms | 18.49 ms | 55.8 | 19,530 |
| **1D-CNN (Spatial/Channel)** | 89.50% | 89.50% | 89.38% | 89.38% | 20.63 ms | 25.81 ms | 48.5 | 35,018 |
| **Residual MLP (Res-Dense)** | **97.00%** | **97.00%** | **96.99%** | **96.99%** | **17.03 ms** | **17.98 ms** | **58.7** | 36,554 |

### Per-Class Performance (Residual MLP Champion)
- Classes `G, I, K, O, P, U, V, X, Y`: 95.0% - 100.0% accuracy.
- Zero $K \leftrightarrow V$ confusion (perfect separation via Level 3 fingertip metrics).
- Minor residual confusion: $S \rightarrow G$ (2 cases due to thumb tuck angle on fist).

## 5. Architectural Decision
**KEEP and ADOPT Residual MLP (Res-Dense).**
- **Decision Rationale:** Outperforms 1D-CNN by +7.50% and standard MLP by +2.00% on unseen test signers.
- Delivers 58.7 FPS inference latency (17.03 ms average), well within the 30 ms real-time requirement.
- Model weights saved at `experiments/exp_003_classifier_comparison/res_mlp_model.keras` and promoted to `models/res_mlp_78d.keras`.
