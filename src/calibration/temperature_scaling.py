"""
Temperature Scaling for Confidence Calibration
Fits optimal scalar temperature T on validation logits using NLL loss.
"""
import numpy as np

class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray, lr=0.01, max_iter=200):
        """
        Optimizes scalar T > 0 on validation set using Gradient Descent on Cross Entropy.
        Does NOT touch test data.
        """
        T = 1.5 # Initial guess
        n = len(val_labels)
        
        for _ in range(max_iter):
            # Scaled logits
            scaled = val_logits / T
            # Softmax
            exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            probs = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
            
            # Cross entropy loss & gradient wrt T
            # grad = -1/T^2 * sum(p_i * z_i - z_y)
            grad = 0.0
            for i in range(n):
                true_c = val_labels[i]
                z = val_logits[i]
                p = probs[i]
                grad += (-1.0 / (T ** 2)) * (np.dot(p, z) - z[true_c])
                
            grad /= n
            T -= lr * grad
            T = max(0.1, min(10.0, T)) # Clamp
            
        self.temperature = float(T)
        return self.temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
