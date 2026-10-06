"""
Train and save the Residual MLP (Res-Dense) champion classifier on 78d features.
Architecture:
- Input(78)
- Dense(128, relu) -> LayerNorm -> Dropout(0.2)
- Residual Block: Dense(128, relu) -> Dropout(0.2) -> Add with previous
- Dense(64, relu) -> Dropout(0.2)
- Dense(10, softmax)
"""
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def train_and_save_res_mlp():
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Load 78d features
    train_data = np.load("real_world_dataset/train_landmarks_78d.npz")
    val_data = np.load("real_world_dataset/validation_landmarks_78d.npz")
    test_data = np.load("real_world_dataset/test_landmarks_78d.npz")
    
    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]
    
    inputs = layers.Input(shape=(78,), name="landmarks_78d")
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Residual block
    res = layers.Dense(128, activation="relu")(x)
    res = layers.Dropout(0.2)(res)
    x = layers.add([x, res])
    x = layers.LayerNormalization()(x)
    
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="ResMLP_78d")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights=True
    )
    
    print("Training Residual MLP...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=70,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy on Locked Unseen Signers: {test_acc * 100:.2f}%")
    
    os.makedirs("experiments/exp_003_classifier_comparison", exist_ok=True)
    out_path = "experiments/exp_003_classifier_comparison/res_mlp_model.keras"
    model.save(out_path)
    print(f"Saved model to {out_path}")
    
    # Also save training script in exp_003 for full reproducibility
    print("Model ready for temporal benchmark.")

if __name__ == "__main__":
    train_and_save_res_mlp()
