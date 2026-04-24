"""
VisionGuard AI — Model Training Script
Run: python backend/train_model.py --data_dir ./dataset --epochs 30
"""

import os
import argparse
import numpy as np
import json
from datetime import datetime

# ─── ARGS ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="./dataset")
    parser.add_argument("--model_dir",  default="./model")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--img_size",   type=int,   default=224)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_split",  type=float, default=0.15)
    return parser.parse_args()

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def train(args):
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Detect classes automatically from subdirectories
    subdirs = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    CLASSES = sorted(subdirs)
    
    if not CLASSES:
        print(f"❌ Error: No subdirectories found in {args.data_dir}")
        print("   Make sure your dataset is structured as: dataset/class_name/images...")
        return

    os.makedirs(args.model_dir, exist_ok=True)

    print("\n" + "="*55)
    print("  VisionGuard AI — Training Started")
    print(f"  Data    : {args.data_dir}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Output  : {args.model_dir}")
    print("="*55 + "\n")

    # ── Data ──
    train_gen = ImageDataGenerator(
        rescale=1./255, validation_split=args.val_split,
        rotation_range=15, width_shift_range=0.1,
        height_shift_range=0.1, zoom_range=0.15,
        horizontal_flip=True, brightness_range=[0.85, 1.15],
    ).flow_from_directory(
        args.data_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size, classes=CLASSES,
        class_mode="categorical", subset="training", seed=42,
    )

    val_gen = ImageDataGenerator(
        rescale=1./255, validation_split=args.val_split,
    ).flow_from_directory(
        args.data_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size, classes=CLASSES,
        class_mode="categorical", subset="validation", seed=42, shuffle=False,
    )

    print(f"  Train: {train_gen.samples} images")
    print(f"  Val  : {val_gen.samples} images\n")

    # Save class map
    class_map = {str(v): k for k, v in train_gen.class_indices.items()}
    with open(os.path.join(args.model_dir, "class_map.json"), "w") as f:
        json.dump(class_map, f, indent=2)

    # ── Model ──
    base = EfficientNetB0(include_top=False, weights="imagenet",
                          input_shape=(args.img_size, args.img_size, 3))
    base.trainable = False

    inputs = tf.keras.Input(shape=(args.img_size, args.img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── Train ──
    ckpt = os.path.join(args.model_dir, "best_model.keras")
    history = model.fit(
        train_gen, validation_data=val_gen,
        epochs=args.epochs,
        callbacks=[
            ModelCheckpoint(ckpt, monitor="val_accuracy", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
            CSVLogger(os.path.join(args.model_dir, "training_log.csv")),
        ],
        verbose=1,
    )

    # ── Save ──
    model.save(os.path.join(args.model_dir, "visionguard_model.keras"))

    # Save history
    with open(os.path.join(args.model_dir, "history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    # ── Plot training curves ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0d1117")
    for ax, (t, v, title) in zip(axes, [
        ("accuracy", "val_accuracy", "Accuracy"),
        ("loss",     "val_loss",     "Loss"),
    ]):
        ax.set_facecolor("#161b22")
        ax.plot(history.history[t], color="#00e5ff", linewidth=2, label="Train")
        ax.plot(history.history[v], color="#00ff88", linewidth=2, label="Val", linestyle="--")
        ax.set_title(title, color="white", fontweight="bold")
        ax.set_xlabel("Epoch", color="#8b949e")
        ax.tick_params(colors="#8b949e")
        ax.legend(facecolor="#21262d", labelcolor="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

    plt.tight_layout()
    plt.savefig(os.path.join(args.model_dir, "training_curves.png"), dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()

    print(f"\n✅ Training complete!")
    print(f"   Model saved → {args.model_dir}/visionguard_model.keras")
    print(f"   Charts saved → {args.model_dir}/training_curves.png\n")


if __name__ == "__main__":
    args = parse_args()
    train(args)
