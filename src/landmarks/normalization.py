"""
Geometric Normalization for Hand Landmarks (21 points x XYZ)
Invariance: Translation-invariant (wrist origin), Scale-invariant (palm span), Rotation-robust.
"""
import numpy as np

def normalize_hand_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalizes a (21, 3) or (N, 21, 3) hand landmarks array.
    Step 1: Translate landmarks so wrist (landmark 0) is the origin (0, 0, 0).
    Step 2: Scale normalize by distance from wrist (point 0) to middle finger MCP (point 9).
    Step 3: Flatten into a 63-dimensional normalized feature vector.
    """
    if landmarks is None or len(landmarks) == 0:
        return np.zeros(63, dtype=np.float32)
        
    pts = np.array(landmarks, dtype=np.float32).copy()
    if pts.shape == (63,):
        pts = pts.reshape(21, 3)
    elif pts.ndim == 2 and pts.shape[1] == 2:
        # Pad with 0 for z if only 2D provided
        pts = np.hstack([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)])
        
    # Wrist as origin
    wrist = pts[0, :].copy()
    pts_centered = pts - wrist
    
    # Scale normalization: distance from wrist (0) to middle MCP (9)
    # If point 9 is degenerate, fallback to maximum Euclidean distance to any landmark
    mcp_dist = np.linalg.norm(pts_centered[9, :])
    if mcp_dist < 1e-6:
        max_dist = np.max(np.linalg.norm(pts_centered, axis=1))
        scale = max_dist if max_dist > 1e-6 else 1.0
    else:
        scale = mcp_dist
        
    pts_normalized = pts_centered / scale
    return pts_normalized.flatten()
