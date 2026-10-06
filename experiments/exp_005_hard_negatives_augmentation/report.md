# Experiment 005: Hard Negative Mining and Geometric Augmentation

## 1. Problem Statement
Examine whether targeted geometric augmentation (Gaussian jitter $\sigma=0.012$, in-plane rotation $\pm 12^\circ$, isotropic scaling $0.94-1.06$, with recomputed Level 3 metrics) on confusion classes ($G, S, K, V$) resolves residual misclassification pairs ($S \leftrightarrow G$ and $K \leftrightarrow V$) without degrading unseen-signer generalization.

## 2. Hypothesis
Over-sampling and perturbing confusing signs in the training split will sharpen decision boundaries and improve accuracy on unseen signers.

## 3. Protocol & Data Safeguards
- **Training Set:** Original 600 samples augmented to 3,120 samples via physiological perturbation with Level 3 distance recomputation.
- **Validation Set:** Unchanged (200 samples from 2 validation signers).
- **Locked Test Set:** Completely untouched (200 samples from signers `signer_09` and `signer_10`).
- **Leakage Verification:** `scripts/verify_no_leakage.py` passed with 0 duplicate SHA-256 signatures and 0 parent capture overlap.

## 4. Empirical Results

| Experiment | Accuracy | Macro F1 | Latency (mean) | P95 Latency | FPS | Test Confusion Errors |
|---|---:|---:|---:|---:|---:|---:|
| **Exp 003: ResMLP (Clean Baseline)** | **97.00%** | **96.99%** | **17.03 ms** | **17.98 ms** | **58.7** | **6 / 200** |
| **Exp 005: ResMLP (Hard Neg Aug)** | 95.50% | 95.52% | 16.97 ms | 17.59 ms | 58.9 | 9 / 200 |

### Error Analysis
- Under aggressive random angular perturbation, the model broadened its decision boundaries around $V$ and $K$, resulting in 3 false positives of $V \rightarrow K$.
- Accuracy dropped by **-1.50%** (from 97.00% down to 95.50%), and confusion errors rose from 6 to 9.

## 5. Engineering Decision
**REJECT aggressive hard-negative geometric augmentation.**
- In accordance with the non-negotiable rule "Measure before/after every change; do not retain changes that degrade unseen-signer performance", this augmentation is rejected.
- The clean training protocol from Experiment 003 with champion weights `experiments/exp_003_classifier_comparison/res_mlp_model.keras` is retained as the production champion model.
