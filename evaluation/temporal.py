"""
Temporal Aggregation Evaluator (Majority Vote, Probability Averaging, EMA)
"""
from collections import Counter
import numpy as np

def evaluate_temporal_aggregation(frame_probs_sequences, sequence_labels, classes):
    """
    Evaluates:
    1. Single Frame (Level 1)
    2. Majority Voting (Level 2)
    3. Probability Averaging
    4. Exponential Moving Average (EMA)
    """
    all_frame_preds = []
    all_frame_trues = []
    
    maj_vote_preds = []
    prob_avg_preds = []
    ema_preds = []
    true_labels = []
    
    for seq_probs, true_label in zip(frame_probs_sequences, sequence_labels):
        true_labels.append(true_label)
        seq_preds = [classes[np.argmax(p)] for p in seq_probs]
        for p in seq_preds:
            all_frame_preds.append(p)
            all_frame_trues.append(true_label)
            
        # Majority Vote
        vote = Counter(seq_preds).most_common(1)[0][0]
        maj_vote_preds.append(vote)
        
        # Prob Avg
        avg_prob = np.mean(seq_probs, axis=0)
        prob_avg_preds.append(classes[np.argmax(avg_prob)])
        
        # EMA (alpha=0.4)
        ema = np.zeros_like(seq_probs[0])
        alpha = 0.4
        for p in seq_probs:
            ema = alpha * p + (1 - alpha) * ema
        ema_preds.append(classes[np.argmax(ema)])
        
    frame_acc = float(np.mean([p == t for p, t in zip(all_frame_preds, all_frame_trues)]))
    maj_acc = float(np.mean([p == t for p, t in zip(maj_vote_preds, true_labels)]))
    prob_acc = float(np.mean([p == t for p, t in zip(prob_avg_preds, true_labels)]))
    ema_acc = float(np.mean([p == t for p, t in zip(ema_preds, true_labels)]))
    
    return {
        'frame_accuracy': frame_acc,
        'majority_vote_accuracy': maj_acc,
        'prob_averaging_accuracy': prob_acc,
        'ema_accuracy': ema_acc,
        'voting_gain_pp': (maj_acc - frame_acc) * 100.0,
        'prob_avg_gain_pp': (prob_acc - frame_acc) * 100.0
    }
