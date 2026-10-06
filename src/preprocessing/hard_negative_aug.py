"""
Hard Negative Mining and Targeted Physiological Augmentation
Applies:
1. Targeted jitter & rotation on confusing class pairs (G, S, K, V)
2. Realistic physiological variations (tremor sigma=0.01-0.03, scale 0.92-1.08, in-plane rotation +/- 15 deg)
3. Recomputes exact Level 3 distances from augmented coordinates
4. STRICT ISOLATION: Applied ONLY to the 600 training samples (NEVER touches validation or locked test set).
"""
import numpy as np

def augment_landmarks_78d(X_train: np.ndarray, y_train: np.ndarray, num_aug_per_sample: int = 3) -> tuple:
    """
    Augments 78d features by:
    1. Extracting 63d normalized coordinates (21 x 3)
    2. Applying random rotation in xy plane, scaling, and Gaussian jitter
    3. Recomputing the 15 Level 3 distances on the perturbed coordinates
    4. Appending to the training set.
    """
    np.random.seed(42)
    classes = ["G", "I", "K", "O", "P", "S", "U", "V", "X", "Y"]
    
    # Indices for the 15 distances:
    # 5 fingertip-to-wrist (tips: 4, 8, 12, 16, 20; wrist: 0)
    # 10 pairwise fingertip distances (4-8, 4-12, 4-16, 4-20, 8-12, 8-16, 8-20, 12-16, 12-20, 16-20)
    tips = [4, 8, 12, 16, 20]
    
    X_aug_list = [X_train]
    y_aug_list = [y_train]
    
    for i in range(len(X_train)):
        feat = X_train[i]
        label = y_train[i]
        c_name = classes[label]
        
        # Give higher augmentation multiplier (x5) to confusion classes (G, S, K, V)
        mult = num_aug_per_sample * 2 if c_name in ["G", "S", "K", "V"] else num_aug_per_sample
        
        coords = feat[:63].reshape(21, 3)
        
        for _ in range(mult):
            # 1. Random in-plane rotation angle [-12 deg, +12 deg]
            theta = np.radians(np.random.uniform(-12.0, 12.0))
            c_cos, s_sin = np.cos(theta), np.sin(theta)
            R = np.array([
                [c_cos, -s_sin, 0],
                [s_sin,  c_cos, 0],
                [0,      0,     1]
            ])
            
            # 2. Random isotropic scale [0.94, 1.06]
            scale = np.random.uniform(0.94, 1.06)
            
            # 3. Random Gaussian tremor jitter (sigma=0.012)
            noise = np.random.normal(0, 0.012, size=coords.shape)
            
            # Transform
            new_coords = (coords @ R.T) * scale + noise
            
            # Recompute 15 distances
            # Wrist distance (landmark 0)
            wrist = new_coords[0]
            wrist_dists = [np.linalg.norm(new_coords[t] - wrist) for t in tips]
            
            # Pairwise fingertip distances
            pair_dists = []
            for idx_a in range(len(tips)):
                for idx_b in range(idx_a + 1, len(tips)):
                    t_a = tips[idx_a]
                    t_b = tips[idx_b]
                    pair_dists.append(np.linalg.norm(new_coords[t_a] - new_coords[t_b]))
                    
            new_feat = np.concatenate([new_coords.flatten(), np.array(wrist_dists), np.array(pair_dists)])
            X_aug_list.append(new_feat.reshape(1, 78))
            y_aug_list.append(np.array([label]))
            
    X_train_augmented = np.vstack(X_aug_list)
    y_train_augmented = np.concatenate(y_aug_list)
    
    return X_train_augmented, y_train_augmented

if __name__ == "__main__":
    train_data = np.load("real_world_dataset/train_landmarks_78d.npz")
    X_tr, y_tr = train_data["X"], train_data["y"]
    print(f"Original Train: X={X_tr.shape}, y={y_tr.shape}")
    
    X_aug, y_aug = augment_landmarks_78d(X_tr, y_tr, num_aug_per_sample=3)
    print(f"Augmented Train: X={X_aug.shape}, y={y_aug.shape}")
    
    np.savez_compressed("real_world_dataset/train_landmarks_78d_augmented.npz", X=X_aug, y=y_aug)
    print("Saved real_world_dataset/train_landmarks_78d_augmented.npz successfully.")
