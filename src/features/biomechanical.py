"""
Biomechanical & Geometric Feature Engineering for Hand Landmarks
Computes:
1. Normalized (x,y,z) coordinates (63 dims)
2. Fingertip-to-Wrist Euclidean Distances (5 dims)
3. Inter-Fingertip Pairwise Distances (10 dims)
4. Finger Extension Ratios (Tip distance vs MCP distance) (5 dims)
5. Inter-Finger Angles at MCP joints (4 dims)
Total Feature Dimension: 63 + 5 + 10 + 5 + 4 = 87 dims
"""
import numpy as np

TIPS = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky
MCPS = [2, 5, 9, 13, 17]
WRIST = 0

def extract_engineered_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Extracts an 87-dimensional feature vector combining normalized coordinates
    and explicit biomechanical finger metrics.
    """
    if landmarks is None or len(landmarks) == 0:
        return np.zeros(87, dtype=np.float32)
        
    pts = np.array(landmarks, dtype=np.float32).copy()
    if pts.shape == (63,):
        pts = pts.reshape(21, 3)
    elif pts.ndim == 2 and pts.shape[1] == 2:
        pts = np.hstack([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)])
        
    # 1. Wrist Centering and Scale Normalization
    wrist = pts[0, :].copy()
    pts_centered = pts - wrist
    
    mcp_dist = np.linalg.norm(pts_centered[9, :])
    scale = mcp_dist if mcp_dist > 1e-6 else 1.0
    norm_pts = pts_centered / scale
    
    coord_feats = norm_pts.flatten() # 63 dims
    
    # 2. Fingertip to Wrist Distances (5 dims)
    tip_wrist_dists = np.array([np.linalg.norm(norm_pts[t]) for t in TIPS], dtype=np.float32)
    
    # 3. Inter-Fingertip Pairwise Distances (10 dims: C(5, 2))
    inter_tip_dists = []
    for i in range(len(TIPS)):
        for j in range(i + 1, len(TIPS)):
            d = np.linalg.norm(norm_pts[TIPS[i]] - norm_pts[TIPS[j]])
            inter_tip_dists.append(d)
    inter_tip_dists = np.array(inter_tip_dists, dtype=np.float32)
    
    # 4. Finger Extension Ratios: ||Tip|| / (||MCP|| + 1e-6) (5 dims)
    extension_ratios = []
    for t, m in zip(TIPS, MCPS):
        d_tip = np.linalg.norm(norm_pts[t])
        d_mcp = np.linalg.norm(norm_pts[m])
        ratio = d_tip / (d_mcp + 1e-5)
        extension_ratios.append(ratio)
    extension_ratios = np.array(extension_ratios, dtype=np.float32)
    
    # 5. Inter-Finger Separation Angles between adjacent MCPs (4 dims)
    angles = []
    for i in range(len(TIPS) - 1):
        v1 = norm_pts[TIPS[i]] - norm_pts[MCPS[i]]
        v2 = norm_pts[TIPS[i+1]] - norm_pts[MCPS[i+1]]
        cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_ang = np.clip(cos_ang, -1.0, 1.0)
        angles.append(cos_ang)
    angles = np.array(angles, dtype=np.float32)
    
    return np.concatenate([coord_feats, tip_wrist_dists, inter_tip_dists, extension_ratios, angles])
