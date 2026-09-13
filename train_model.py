# ============================================================
# REAL-TIME FACE MASK DETECTION USING CNN
# MODEL TRAINING SCRIPT
# ============================================================

import os
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset"

MODEL_PATH = "mask_detector.keras"

CLASS_NAMES_PATH = "class_names.json"

IMG_SIZE = 128

BATCH_SIZE = 32

EPOCHS = 15

LEARNING_RATE = 0.0001

SEED = 42


# ============================================================
# START MESSAGE
# ============================================================

print("\n")
print("==========================================")
print("   FACE MASK DETECTION - MODEL TRAINING")
print("==========================================")
print("\n")


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_DIR):

    raise FileNotFoundError(
        f"Dataset folder '{DATASET_DIR}' was not found."
    )


print("Dataset directory:")
print(os.path.abspath(DATASET_DIR))


# ============================================================
# DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    validation_split=0.2,

    rotation_range=20,

    width_shift_range=0.1,

    height_shift_range=0.1,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True
)


# ============================================================
# VALIDATION DATA GENERATOR
# ============================================================

validation_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    validation_split=0.2
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training dataset...\n")


train_data = train_datagen.flow_from_directory(

    DATASET_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="training",

    shuffle=True,

    seed=SEED
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation dataset...\n")


val_data = validation_datagen.flow_from_directory(

    DATASET_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="validation",

    shuffle=False,

    seed=SEED
)


# ============================================================
# DISPLAY CLASS MAPPING
# ============================================================

print("\n==========================================")
print("CLASS MAPPING")
print("==========================================")

print(train_data.class_indices)


# Example:
# {'with_mask': 0, 'without_mask': 1}


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_names = {
    str(value): key
    for key, value in train_data.class_indices.items()
}


with open(
    CLASS_NAMES_PATH,
    "w"
) as file:

    json.dump(
        class_names,
        file,
        indent=4
    )


print("\nClass names saved as:")
print(CLASS_NAMES_PATH)

print("\nClass mapping:")
print(class_names)


# ============================================================
# BUILD MOBILE NET V2
# ============================================================

print("\n==========================================")
print("BUILDING MODEL")
print("==========================================\n")


# Load MobileNetV2 without ImageNet classification layer

base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)


# Freeze pretrained layers

base_model.trainable = False


# ============================================================
# CREATE CLASSIFICATION HEAD
# ============================================================

inputs = Input(
    shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)


x = base_model(
    inputs,
    training=False
)


x = GlobalAveragePooling2D()(x)


x = Dense(
    128,
    activation="relu"
)(x)


x = Dropout(
    0.4
)(x)


outputs = Dense(
    1,
    activation="sigmoid"
)(x)


model = Model(
    inputs,
    outputs
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


# ============================================================
# DISPLAY MODEL
# ============================================================

print("\nModel architecture:\n")

model.summary()


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n==========================================")
print("STARTING TRAINING")
print("==========================================\n")


history = model.fit(

    train_data,

    validation_data=val_data,

    epochs=EPOCHS
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    MODEL_PATH
)


print("\n==========================================")
print("MODEL SAVED")
print("==========================================")

print(
    f"\nModel saved successfully as: {MODEL_PATH}"
)


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n")
print("==========================================")
print("MODEL EVALUATION")
print("==========================================\n")


evaluation = model.evaluate(
    val_data,
    verbose=1
)


print("\nEvaluation results:")

for metric_name, metric_value in zip(
    model.metrics_names,
    evaluation
):

    print(
        f"{metric_name}: {metric_value:.4f}"
    )


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...\n")


val_data.reset()


predictions = model.predict(
    val_data,
    verbose=1
)


# Convert probabilities into classes

predicted_classes = (
    predictions.ravel() >= 0.5
).astype(int)


true_classes = val_data.classes


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("==========================================")
print("CLASSIFICATION REPORT")
print("==========================================\n")


target_names = [
    class_names[str(i)]
    for i in range(len(class_names))
]


report = classification_report(

    true_classes,

    predicted_classes,

    target_names=target_names,

    digits=4
)


print(report)


# Save classification report

with open(
    "classification_report.txt",
    "w"
) as file:

    file.write(report)


print(
    "Classification report saved as: "
    "classification_report.txt"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    true_classes,

    predicted_classes
)


print("\n")
print("==========================================")
print("CONFUSION MATRIX")
print("==========================================\n")


print(cm)


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=target_names
)


fig, ax = plt.subplots(
    figsize=(7, 7)
)


disp.plot(
    ax=ax
)


plt.title(
    "Face Mask Detection - Confusion Matrix"
)


plt.tight_layout()


plt.savefig(

    "confusion_matrix.png",

    dpi=300,

    bbox_inches="tight"
)


plt.close(fig)


print(
    "\nConfusion matrix saved as: "
    "confusion_matrix.png"
)


# ============================================================
# TRAINING ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    history.history["accuracy"],

    label="Training Accuracy"
)


plt.plot(

    history.history["val_accuracy"],

    label="Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.title(
    "Training vs Validation Accuracy"
)


plt.legend()


plt.grid()


plt.tight_layout()


plt.savefig(

    "accuracy_curve.png",

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print(
    "Accuracy graph saved as: "
    "accuracy_curve.png"
)


# ============================================================
# TRAINING LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    history.history["loss"],

    label="Training Loss"
)


plt.plot(

    history.history["val_loss"],

    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "Training vs Validation Loss"
)


plt.legend()


plt.grid()


plt.tight_layout()


plt.savefig(

    "loss_curve.png",

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print(
    "Loss graph saved as: "
    "loss_curve.png"
)


# ============================================================
# PRECISION GRAPH
# ============================================================

if "precision" in history.history:

    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(

        history.history["precision"],

        label="Training Precision"
    )


    plt.plot(

        history.history["val_precision"],

        label="Validation Precision"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Precision"
    )


    plt.title(
        "Training vs Validation Precision"
    )


    plt.legend()


    plt.grid()


    plt.tight_layout()


    plt.savefig(

        "precision_curve.png",

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()


    print(
        "Precision graph saved as: "
        "precision_curve.png"
    )


# ============================================================
# RECALL GRAPH
# ============================================================

if "recall" in history.history:

    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(

        history.history["recall"],

        label="Training Recall"
    )


    plt.plot(

        history.history["val_recall"],

        label="Validation Recall"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Recall"
    )


    plt.title(
        "Training vs Validation Recall"
    )


    plt.legend()


    plt.grid()


    plt.tight_layout()


    plt.savefig(

        "recall_curve.png",

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()


    print(
        "Recall graph saved as: "
        "recall_curve.png"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("==========================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("==========================================\n")


print(
    f"Model: {MODEL_PATH}"
)

print(
    f"Class mapping: {CLASS_NAMES_PATH}"
)

print(
    "Classification report: "
    "classification_report.txt"
)

print(
    "Confusion matrix: "
    "confusion_matrix.png"
)

print(
    "Accuracy graph: "
    "accuracy_curve.png"
)

print(
    "Loss graph: "
    "loss_curve.png"
)

print(
    "Precision graph: "
    "precision_curve.png"
)

print(
    "Recall graph: "
    "recall_curve.png"
)


print("\n==========================================")
print("NEXT STEP")
print("==========================================\n")

print(
    "Run the real-time detector using:"
)

print(
    "python realtime_detection.py"
)

print("\n")