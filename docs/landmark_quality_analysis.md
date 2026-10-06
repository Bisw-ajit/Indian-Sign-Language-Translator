# Landmark Quality Audit: Indian Sign Language Recognition

**Audit Date**: October 2026  
**Auditor**: ML & Computer Vision Research Team  
**Investigative Focus**: Information sufficiency, geometric integrity, and dimensional capacity of the 21 MediaPipe hand landmarks.

---

## 1. Landmark Structural Integrity & Geometric Validity

MediaPipe Hands outputs 21 anatomical landmarks per detected hand, representing:
- **Wrist**: Landmark 0 (Anatomical anchor)
- **Thumb**: Landmarks 1, 2, 3, 4 (CMC, MCP, IP, Tip)
- **Index Finger**: Landmarks 5, 6, 7, 8 (MCP, PIP, DIP, Tip)
- **Middle Finger**: Landmarks 9, 10, 11, 12 (MCP, PIP, DIP, Tip)
- **Ring Finger**: Landmarks 13, 14, 15, 16 (MCP, PIP, DIP, Tip)
- **Pinky Finger**: Landmarks 17, 18, 19, 20 (MCP, PIP, DIP, Tip)

### Measured Geometric Fidelity:
- On real physical captures (`extracted_thesis_media`), whenever a human hand is detected, **all 21 landmarks are successfully extracted with 100% geometric completeness** (`valid_landmarks = 21/21`).
- Mean detection confidence on valid human gestures is exceptionally high: **0.88 to 0.99**.
- The 21 landmarks successfully capture finger curling, extension, and thumb abduction across natural human hands.

---

## 2. Identified Representation Gaps

Despite the anatomical precision of individual hands, three critical representation bottlenecks were identified:

1. **Two-Handed Occlusion & Handedness Truncation**:
   - In ISL gestures `G`, `K`, `S`, both hands interact closely.
   - For `G` (stacked fists), MediaPipe detects **2 hands** (Hand 0 conf 0.89, Hand 1 conf 0.88).
   - *Current Limitation*: The baseline landmark extractor truncated the input to `res[0]`, completely discarding the secondary hand. This renders two-handed signs (`G`, `S`) indistinguishable from single-handed signs.
   
2. **Lack of Explicit Angular & Distance Features**:
   - Raw normalized Cartesian coordinates $(x_i, y_i, z_i)$ require neural networks to implicitly learn trigonometric angles and Euclidean metrics.
   - For subtle distinctions (e.g. `U` where index and middle fingers are parallel and touching vs `V` where they are splayed apart), fingertip Euclidean separation `||tip_index - tip_middle||` is the definitive discriminative signal.

3. **Multi-Hand Extension Specification**:
   - Expanding the landmark representation to support dual hands ($2 \times 21 \times 3 = 126$ dimensions) with relative inter-wrist positioning will preserve the full sign language grammar.
