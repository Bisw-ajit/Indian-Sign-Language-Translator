"""
Confusion Matrix Visualization and Export
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def save_confusion_matrix(cm, classes, output_csv_path, output_png_path, title='Confusion Matrix'):
    """
    Saves confusion matrix to CSV and creates a publication-quality PNG plot.
    """
    # Write CSV
    with open(output_csv_path, 'w') as f:
        f.write("," + ",".join(classes) + "\n")
        for idx, row in enumerate(cm):
            f.write(f"{classes[idx]}," + ",".join(map(str, row)) + "\n")
            
    # Plot PNG
    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    n_cls = len(classes)
    ax.set_xticks(range(n_cls))
    ax.set_yticks(range(n_cls))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.xlabel('Predicted Class', fontweight='bold')
    plt.ylabel('Ground Truth Class', fontweight='bold')
    plt.title(title, pad=20, fontweight='bold')
    
    for i in range(n_cls):
        for j in range(n_cls):
            val = cm[i, j]
            color = "white" if val > (cm.max() / 2) else "black"
            ax.text(j, i, str(val), va='center', ha='center', color=color, fontsize=10)
            
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=200)
    plt.close()
