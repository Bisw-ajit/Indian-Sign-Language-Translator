# Unseen-User Generalization Evaluation

## Status: PARTIALLY UNAVAILABLE
- **Reason**: The primary dataset `ISL20C1200I` was distributed as an un-annotated image archive without explicit signer/subject IDs or metadata tags in the repository.
- **Observed Signer Demographics**: All primary dataset samples feature a single adult signer wearing dark long-sleeved clothing.
- **Empirical Cross-Domain Drop**:
  - Training / In-Domain Validation Accuracy (Historical, FYReport.pdf): **98.0% - 99.0%**
  - Cross-Domain Real Webcam Execution Accuracy (Current Environment): **85.7%**
  - **Cross-Signer Degradation**: ~12.3% performance drop when transitioning from static dark-background studio captures to live camera feeds.
