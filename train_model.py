import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    Callback,
)
import tensorflow as tf
import json
import warnings
import time
import platform
import multiprocessing
from datetime import timedelta

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class EpochProgressCallback(Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
        print(f"\n----- Epoch {epoch + 1}/{self.params['epochs']} -----")

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        total_epochs = self.params["epochs"]
        elapsed = time.time() - self.start_time
        remaining_epochs = total_epochs - (epoch + 1)
        eta = timedelta(seconds=int((elapsed / (epoch + 1)) * remaining_epochs))

        print(
            f"Time for this epoch: {epoch_time:.1f}s | "
            f"ETA for remaining: {eta} | "
            f"Val Accuracy: {logs.get('val_accuracy'):.4f}"
        )


def main():
    num_cores = multiprocessing.cpu_count()
    tf.config.threading.set_intra_op_parallelism_threads(num_cores)
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    tf.config.set_soft_device_placement(True)

    print("\n" + "=" * 70)
    print("MediScan AI - Optimized CPU Training Mode")
    print("=" * 70 + "\n")

    print(f"System: {platform.system()}")
    print(f"CPU Cores Available: {num_cores}")
    print(f"Using TensorFlow Threads: {num_cores}")
    print("GPU Disabled - Full CPU Utilization Enabled\n")

    BASE_DIR = os.path.join("chest_xray", "chest_xray")
    TRAIN_DIR = os.path.join(BASE_DIR, "train")
    VAL_DIR = os.path.join(BASE_DIR, "val")
    TEST_DIR = os.path.join(BASE_DIR, "test")

    if not os.path.exists(VAL_DIR):
        VAL_DIR = TEST_DIR

    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 10 
    LEARNING_RATE = 0.001
    MODEL_PATH = "pneumonia_model.h5"

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    print("Loading datasets...")

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    print(f"\nClass mapping: {train_generator.class_indices}\n")

    print("Building model...")

    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(2, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, epsilon=1e-7),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"Model ready: {model.count_params():,} parameters")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Device: CPU ({num_cores} cores)\n")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        ),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
        EpochProgressCallback(), 
    ]

    print("Starting CPU training...\n")
    start_time = time.time()

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator),
        workers=1,
        use_multiprocessing=False,
    )

    print("\nEvaluating model on test set...")
    test_loss, test_acc = model.evaluate(test_generator, verbose=1)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    print(f"\nSaving model as '{MODEL_PATH}'...")
    model.save(MODEL_PATH)
    model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"Saved ({model_size:.2f} MB)\n")

    history_data = {
        "accuracy": [float(x) for x in history.history["accuracy"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
        "loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "epochs_trained": len(history.history["accuracy"]),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "cpu_cores": num_cores,
        "total_training_time_minutes": (time.time() - start_time) / 60,
    }

    with open("training_history.json", "w") as f:
        json.dump(history_data, f, indent=4)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"], label="Train Accuracy", marker="o")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", marker="s")
    axes[0].set_title("Model Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Train Loss", marker="o")
    axes[1].plot(history.history["val_loss"], label="Val Loss", marker="s")
    axes[1].set_title("Model Loss")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150)
    plt.close()

    total_time = (time.time() - start_time) / 60
    print("=" * 78)
    print("TRAINING COMPLETE (CPU Optimized)")
    print("=" * 78)
    print(f"CPU Cores Used: {num_cores}")
    print(f"Best Val Accuracy: {max(history.history['val_accuracy']) * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Total Time: {total_time:.2f} min")
    print(f"Model: {MODEL_PATH} ({model_size:.2f} MB)")
    print(f"Graph: training_results.png")
    print("=" * 78)


if __name__ == "__main__":
    main()
