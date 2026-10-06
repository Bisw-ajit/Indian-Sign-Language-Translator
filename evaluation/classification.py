"""
Classification Metrics Evaluator - Zero Fabrication & Mathematically Exact
"""
import math
import numpy as np

def compute_classification_metrics(y_true, y_pred, classes):
    """
    Computes accuracy, balanced accuracy, precision, recall, F1 (macro & weighted),
    Cohen's kappa, and MCC.
    """
    n_cls = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true_idx = [class_to_idx[y] if isinstance(y, str) else y for y in y_true]
    y_pred_idx = [class_to_idx[y] if isinstance(y, str) else y for y in y_pred]
    
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true_idx, y_pred_idx):
        cm[t, p] += 1
        
    total = int(np.sum(cm))
    acc = float(np.trace(cm) / total) if total > 0 else 0.0
    
    supports = [int(np.sum(cm[i, :])) for i in range(n_cls)]
    precisions = [float(cm[i, i] / np.sum(cm[:, i])) if np.sum(cm[:, i]) > 0 else 0.0 for i in range(n_cls)]
    recalls = [float(cm[i, i] / np.sum(cm[i, :])) if np.sum(cm[i, :]) > 0 else 0.0 for i in range(n_cls)]
    f1s = [float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0 for p, r in zip(precisions, recalls)]
    
    bal_acc = float(np.mean(recalls))
    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    
    weighted_prec = float(sum(p * s for p, s in zip(precisions, supports)) / total) if total > 0 else 0.0
    weighted_rec = float(sum(r * s for r, s in zip(recalls, supports)) / total) if total > 0 else 0.0
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / total) if total > 0 else 0.0
    
    # Cohen's Kappa
    po = acc
    pe = float(sum(np.sum(cm[i, :]) * np.sum(cm[:, i]) for i in range(n_cls)) / (total ** 2)) if total > 0 else 0.0
    kappa = float((po - pe) / (1 - pe)) if (1 - pe) != 0 else 0.0
    
    # Matthews Correlation Coefficient
    c = float(np.trace(cm))
    s = float(total)
    p_vec = np.sum(cm, axis=0)
    t_vec = np.sum(cm, axis=1)
    cov = c * s - float(np.dot(p_vec, t_vec))
    denom = math.sqrt(max(0.0, (s**2 - float(np.dot(p_vec, p_vec))) * (s**2 - float(np.dot(t_vec, t_vec)))))
    mcc = float(cov / denom) if denom > 0 else 0.0
    
    per_class = {}
    for i, cls in enumerate(classes):
        tp = int(cm[i, i])
        fn = int(np.sum(cm[i, :]) - tp)
        fp = int(np.sum(cm[:, i]) - tp)
        tn = int(total - (tp + fn + fp))
        per_class[cls] = {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'precision': precisions[i],
            'recall': recalls[i],
            'f1': f1s[i],
            'support': supports[i]
        }
        
    return {
        'confusion_matrix': cm,
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_prec,
        'weighted_recall': weighted_rec,
        'weighted_f1': weighted_f1,
        'cohen_kappa': kappa,
        'mcc': mcc,
        'per_class': per_class,
        'total_samples': total
    }
