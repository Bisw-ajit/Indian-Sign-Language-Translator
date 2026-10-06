"""
Confidence Calibration Metrics (ECE, Brier Score, NLL, Reliability Diagrams)
"""
import numpy as np

def compute_calibration_metrics(y_true_indices, y_probs, n_bins=10):
    """
    Computes Expected Calibration Error (ECE), Multi-Class Brier Score, and NLL.
    """
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true_indices).astype(float)
    
    n_samples = len(y_true_indices)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    bin_data = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_data.append({
                'bin': i,
                'range': (bin_lower, bin_upper),
                'accuracy': float(accuracy_in_bin),
                'confidence': float(avg_confidence_in_bin),
                'weight': float(prop_in_bin),
                'count': int(np.sum(in_bin))
            })
            
    # Multi-class Brier Score
    n_classes = y_probs.shape[1]
    one_hot = np.zeros((n_samples, n_classes))
    for idx, true_c in enumerate(y_true_indices):
        one_hot[idx, true_c] = 1.0
    brier_score = float(np.mean(np.sum((y_probs - one_hot) ** 2, axis=1)))
    
    # NLL
    eps = 1e-12
    nll = float(-np.mean(np.log(y_probs[np.arange(n_samples), y_true_indices] + eps)))
    
    correct_mask = (predictions == y_true_indices)
    mean_conf_correct = float(np.mean(confidences[correct_mask])) if np.any(correct_mask) else 0.0
    mean_conf_incorrect = float(np.mean(confidences[~correct_mask])) if np.any(~correct_mask) else 0.0
    
    return {
        'ece': float(ece),
        'brier_score': brier_score,
        'nll': nll,
        'mean_confidence_correct': mean_conf_correct,
        'mean_confidence_incorrect': mean_conf_incorrect,
        'bin_data': bin_data
    }
