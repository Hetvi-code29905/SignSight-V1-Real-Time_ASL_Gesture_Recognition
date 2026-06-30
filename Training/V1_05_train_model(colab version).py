import numpy as np
import json
import argparse
import sys
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

if "google.colab" in sys.modules or "IPython" in sys.modules:
    sys.path.insert(0, "/content/drive/MyDrive/SignSight_AI")
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DATA_DIR, MODEL_DIR, LOG_DIR, CHECKPOINT_DIR,
    SEQUENCE_LENGTH, FEATURE_DIM,
    LSTM_UNITS_1, LSTM_UNITS_2, DENSE_UNITS, DROPOUT_RATE,
    LEARNING_RATE, BATCH_SIZE, EPOCHS, PATIENCE
)

def build_bilstm_model(input_shape: tuple, num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="landmark_sequence")
    
    x = layers.Bidirectional(
        layers.LSTM(
            LSTM_UNITS_1,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.0,
            kernel_regularizer=regularizers.l2(1e-4),
            name="lstm_1"
        ),
        name="bilstm_1"
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    x = layers.Bidirectional(
        layers.LSTM(
            LSTM_UNITS_1,
            return_sequences=False,
            dropout=0.3,
            recurrent_dropout=0.0,
            kernel_regularizer=regularizers.l2(1e-4),
            name="lstm_2"
        ),
        name="bilstm_2"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    x = layers.Dense(
        DENSE_UNITS,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_1"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE * 0.75)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="SignSight_BiLSTM")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")]
    )
    
    return model

def train(epochs: int, batch_size: int):
    print("=" * 60)
    print("  SignSight AI — BiLSTM Training")
    print("=" * 60)
    
    print("\n  Loading dataset...")
    
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_val   = np.load(DATA_DIR / "X_val.npy")
    y_val   = np.load(DATA_DIR / "y_val.npy")
    
    with open(DATA_DIR / "label_map.json", "r") as f:
        label_info = json.load(f)
    
    num_classes = label_info["num_classes"]
    
    print(f"  Train: {X_train.shape} | Val: {X_val.shape}")
    print(f"  Classes: {num_classes}")
    print(f"  Feature dim: {FEATURE_DIM}")
    
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    model = build_bilstm_model(
        input_shape=(SEQUENCE_LENGTH, FEATURE_DIM),
        num_classes=num_classes
    )
    model.summary()
    
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(CHECKPOINT_DIR / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=PATIENCE // 2,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=str(LOG_DIR / "tensorboard"),
            histogram_freq=1
        )
    ]
    
    print(f"\n  Training for up to {epochs} epochs...")
    print(f"  Batch size: {batch_size}\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    model_path = MODEL_DIR / "signsight_bilstm.keras"
    model.save(str(model_path))
    print(f"\n  ✅ Model saved → {model_path}")
    
    history_data = history.history
    history_path = LOG_DIR / "training_history.json"
    
    serializable = {k: [float(v) for v in vals] for k, vals in history_data.items()}
    with open(history_path, "w") as f:
        json.dump(serializable, f, indent=2)
    
    plot_training_history(history_data)
    
    best_val_acc = max(history_data["val_accuracy"])
    best_epoch   = history_data["val_accuracy"].index(best_val_acc) + 1
    
    print(f"\n{'─'*60}")
    print(f"  Best Val Accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"\n  NEXT STEP:")
    print(f"  Run: python 05_evaluate_model.py")
    print("=" * 60 + "\n")
    
    return history

def plot_training_history(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history["accuracy"],     label="Train Accuracy", color="#6C63FF")
    axes[0].plot(history["val_accuracy"], label="Val Accuracy",   color="#FF6584", linestyle="--")
    axes[0].set_title("Model Accuracy",  fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(history["loss"],     label="Train Loss", color="#6C63FF")
    axes[1].plot(history["val_loss"], label="Val Loss",   color="#FF6584", linestyle="--")
    axes[1].set_title("Model Loss",   fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.suptitle("SignSight AI — BiLSTM Training History", fontsize=16, fontweight="bold")
    plt.tight_layout()
    
    plot_path = LOG_DIR / "training_history.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  📊 Training plot saved → {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SignSight BiLSTM")
    parser.add_argument("--epochs", type=int, default=EPOCHS,      help="Max training epochs")
    parser.add_argument("--batch",  type=int, default=BATCH_SIZE,  help="Batch size")
    args, _ = parser.parse_known_args()
    
    if not (DATA_DIR / "X_train.npy").exists():
        print("[ERROR] Training data not found. Run 03_build_dataset.py first.")
        sys.exit(1)
    
    train(epochs=args.epochs, batch_size=args.batch)
