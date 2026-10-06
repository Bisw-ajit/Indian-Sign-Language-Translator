"""
Experiment 005: Retraining Residual MLP with Hard Negative Augmentation
Compares:
1. ResMLP without Augmentation (Baseline: Exp 003)
2. ResMLP with Hard Negative Augmentation (Exp 005)

Evaluates on the exact same locked held-out test set (unseen signers).
"""
import os
import time
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def compute_metrics(y_true, y_pred, num_classes=10):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float(np.mean(y_true == y_pred))
    
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * (prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return acc, float(np.mean(f1s))

def build_res_mlp():
    inputs = layers.Input(shape=(78,), name="landmarks_78d")
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    res = layers.Dense(128, activation="relu")(x)
    res = layers.Dropout(0.2)(res)
    x = layers.add([x, res])
    x = layers.LayerNormalization()(x)
    
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="ResMLP_78d_Aug")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def run_exp_005():
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # 1. Load Data
    aug_train = np.load("real_world_dataset/train_landmarks_78d_augmented.npz")
    val_data = np.load("real_world_dataset/validation_landmarks_78d.npz")
    test_data = np.load("real_world_dataset/test_landmarks_78d.npz")
    
    X_train_aug, y_train_aug = aug_train["X"], aug_train["y"]
    X_val, y_val = val_data["X"], val_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]
    
    print(f"Training ResMLP on Augmented Set: {X_train_aug.shape}...")
    
    model = build_res_mlp()
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=14, restore_best_weights=True
    )
    
    t0 = time.time()
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    train_time = time.time() - t0
    
    # Evaluate Latency
    latencies = []
    for _ in range(50):
        t_start = time.perf_counter()
        _ = model.predict(X_test[:1], verbose=0)
        latencies.append((time.perf_counter() - t_start) * 1000.0)
    avg_lat = float(np.mean(latencies[5:]))
    p95_lat = float(np.percentile(latencies[5:], 95))
    fps = float(1000.0 / avg_lat)
    
    # Predictions
    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=-1)
    
    acc, macro_f1 = compute_metrics(y_test, preds)
    acc *= 100.0
    macro_f1 *= 100.0
    
    # Confusion matrix
    classes = ["G", "I", "K", "O", "P", "S", "U", "V", "X", "Y"]
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_test, preds):
        cm[t, p] += 1
        
    print(f"\n=== RESULTS (EXP 005: HARD NEGATIVE AUGMENTATION) ===")
    print(f"Test Accuracy: {acc:.2f}% (Macro F1: {macro_f1:.2f}%)")
    print(f"Inference Latency: {avg_lat:.2f} ms | FPS: {fps:.1f}")
    print("Confusion Matrix:")
    print("   " + " ".join(f"{c:>3}" for c in classes))
    for i, row in enumerate(cm):
        print(f"{classes[i]:>2}: " + " ".join(f"{v:>3}" for v in row))
        
    os.makedirs("experiments/exp_005_hard_negatives_augmentation", exist_ok=True)
    out_model = "experiments/exp_005_hard_negatives_augmentation/res_mlp_augmented.keras"
    model.save(out_model)
    
    results = [
        {
            "Experiment": "Exp 003: ResMLP (Clean Train)",
            "Accuracy": 97.0,
            "Macro_F1": 96.99,
            "Latency_ms": 17.03,
            "P95_Latency_ms": 17.98,
            "FPS": 58.7,
            "Confusion_Errors": 6
        },
        {
            "Experiment": "Exp 005: ResMLP (Hard Negatives Aug)",
            "Accuracy": round(acc, 2),
            "Macro_F1": round(macro_f1, 2),
            "Latency_ms": round(avg_lat, 2),
            "P95_Latency_ms": round(p95_lat, 2),
            "FPS": round(fps, 1),
            "Confusion_Errors": int(np.sum(preds != y_test))
        }
    ]
    
    csv_path = "experiments/exp_005_hard_negatives_augmentation/benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Benchmark saved to {csv_path}")

if __name__ == "__main__":
    run_exp_005()
