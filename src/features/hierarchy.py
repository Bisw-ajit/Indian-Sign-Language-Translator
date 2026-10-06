"""
Multi-Level Feature Hierarchy Extractor
Supports:
Level 1: Raw Normalized Coordinates (63 dims)
Level 2: Coordinates + Joint Angles (63 + 4 = 67 dims)
Level 3: Coordinates + Inter-Fingertip Distances (63 + 15 = 78 dims)
Level 4: Coordinates + Angles + Distances + Orientation (63 + 4 + 15 + 4 = 86 dims)
"""
import numpy as np

TIPS = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky
MCPS = [2, 5, 9, 13, 17]
WRIST = 0

def extract_feature_levels(landmarks_63: np.ndarray) -> dict:
    """
    Extracts all 4 representation levels from a normalized 63-dim landmark vector.
    """
    pts = landmarks_63.reshape(21, 3).astype(np.float32)
    
    # Level 1: Coordinates only
    f_coords = landmarks_63.copy() # 63 dims
    
    # 1. Joint angles between adjacent fingers (4 dims)
    # Cosine angle between finger bones (MCP to TIP vectors)
    angles = []
    for i in range(len(TIPS) - 1):
        v1 = pts[TIPS[i]] - pts[MCPS[i]]
        v2 = pts[TIPS[i+1]] - pts[MCPS[i+1]]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        cos_ang = np.dot(v1, v2) / (n1 * n2 + 1e-6)
        angles.append(np.clip(cos_ang, -1.0, 1.0))
    f_angles = np.array(angles, dtype=np.float32) # 4 dims
    
    # 2. Pairwise distances between all fingertips and wrist (10 pairwise tips + 5 to wrist = 15 dims)
    dists = []
    # 10 pairwise fingertip distances
    for i in range(len(TIPS)):
        for j in range(i + 1, len(TIPS)):
            dists.append(np.linalg.norm(pts[TIPS[i]] - pts[TIPS[j]]))
    # 5 fingertip to wrist distances
    for t in TIPS:
        dists.append(np.linalg.norm(pts[t] - pts[WRIST]))
    f_dists = np.array(dists, dtype=np.float32) # 15 dims
    
    # 3. Global Palm Orientation features (4 dims)
    # Normal vector to palm (cross product of wrist->index_mcp and wrist->pinky_mcp)
    v_index = pts[5] - pts[0]
    v_pinky = pts[17] - pts[0]
    palm_normal = np.cross(v_index, v_pinky)
    norm_mag = np.linalg.norm(palm_normal)
    palm_normal = palm_normal / (norm_mag + 1e-6) # 3 dims
    # Hand aspect ratio: width (index_mcp to pinky_mcp) / height (wrist to middle_mcp)
    palm_width = np.linalg.norm(pts[5] - pts[17])
    palm_height = np.linalg.norm(pts[9] - pts[0])
    aspect_ratio = np.array([palm_width / (palm_height + 1e-6)], dtype=np.float32) # 1 dim
    f_orientation = np.concatenate([palm_normal, aspect_ratio]) # 4 dims
    
    # Combinations
    level_1 = f_coords # 63
    level_2 = np.concatenate([f_coords, f_angles]) # 67
    level_3 = np.concatenate([f_coords, f_dists]) # 78
    level_4 = np.concatenate([f_coords, f_angles, f_dists, f_orientation]) # 86
    
    return {
        'coords': level_1,
        'coords_angles': level_2,
        'coords_dists': level_3,
        'coords_angles_dists_orient': level_4
    }
