"""Train a face-mask classifier (with_mask / without_mask) using MobileNetV2 transfer learning.

Usage:
    python train.py --dataset dataset --epochs 20
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224
CLASSES = ["with_mask", "without_mask"]


def load_dataset(root):
    paths, labels = [], []
    for idx, cls in enumerate(CLASSES):
        folder = os.path.join(root, cls)
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(folder, name))
                labels.append(idx)
    return np.array(paths), np.array(labels)


def decode(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img, label


def make_ds(paths, labels, batch, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=42)
    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def build_model():
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.06),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
    ], name="augmentation")

    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False  # freeze the pretrained feature extractor

    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)
    x = base(x, training=False)
    x = layers.AveragePooling2D(pool_size=(7, 7))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)
    return models.Model(inputs, outputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--model", default="models/mask_detector.keras")
    ap.add_argument("--plot", default="models/training_plot.png")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    paths, labels = load_dataset(args.dataset)
    print(f"[INFO] {len(paths)} images: " +
          ", ".join(f"{c}={int((labels == i).sum())}" for i, c in enumerate(CLASSES)))

    x_train, x_test, y_train, y_test = train_test_split(
        paths, labels, test_size=0.20, stratify=labels, random_state=42)
    train_ds = make_ds(x_train, y_train, args.batch, training=True)
    test_ds = make_ds(x_test, y_test, args.batch, training=False)

    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary(line_length=100)

    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.model, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ]
    history = model.fit(train_ds, validation_data=test_ds, epochs=args.epochs, callbacks=callbacks)

    print("[INFO] evaluating on held-out test set...")
    probs = model.predict(test_ds)
    preds = probs.argmax(axis=1)
    print(classification_report(y_test, preds, target_names=CLASSES, digits=4))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_test, preds))

    model.save(args.model)
    print(f"[INFO] model saved to {args.model}")

    h = history.history
    n = range(1, len(h["loss"]) + 1)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(n, h["loss"], label="train_loss")
    plt.plot(n, h["val_loss"], label="val_loss")
    plt.plot(n, h["accuracy"], label="train_acc")
    plt.plot(n, h["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.legend(loc="center right")
    plt.savefig(args.plot)
    print(f"[INFO] training plot saved to {args.plot}")


if __name__ == "__main__":
    main()
