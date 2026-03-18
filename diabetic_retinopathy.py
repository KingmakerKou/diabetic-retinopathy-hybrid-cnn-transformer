Python 3.11.2 (v3.11.2:878ead1ac1, Feb  7 2023, 10:02:41) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> # =========================
... # IMPORTS
... # =========================
... import tensorflow as tf
... import pandas as pd
... import numpy as np
... import matplotlib.pyplot as plt
... import seaborn as sns
... 
... from tensorflow.keras.applications import DenseNet121
... from tensorflow.keras.layers import (
...     Input, Dense, GlobalAveragePooling2D,
...     MultiHeadAttention, LayerNormalization,
...     Add, Reshape
... )
... from tensorflow.keras.models import Model
... from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
... from tensorflow.keras.preprocessing.image import ImageDataGenerator
... 
... from sklearn.metrics import confusion_matrix, classification_report
... 
# =========================
# PATHS & DATA
# =========================
APTOS_PATH = "/kaggle/input/aptos2019"

train_df = pd.read_csv(f"{APTOS_PATH}/train_1.csv")
val_df   = pd.read_csv(f"{APTOS_PATH}/valid.csv")

# Fix filenames + labels
for df in [train_df, val_df]:
    df["id_code"] = df["id_code"].astype(str).str.replace(".png", "", regex=False) + ".png"
    df["diagnosis"] = df["diagnosis"].astype(str)

# =========================
# DATA PIPELINE
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5

train_dir = f"{APTOS_PATH}/train_images/train_images"
val_dir   = f"{APTOS_PATH}/val_images/val_images"

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col="id_code",
    y_col="diagnosis",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=True
)

val_data = val_gen.flow_from_dataframe(
    dataframe=val_df,
    directory=val_dir,
    x_col="id_code",
    y_col="diagnosis",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=False
)

# =========================
# MODEL DEFINITION
# =========================
def hybrid_densenet_transformer():
    inputs = Input(shape=(224, 224, 3))

    base_model = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)

    # Transformer-style attention
    x = Reshape((1, 512))(x)
    attn = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)

    x = Reshape((512,))(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    return Model(inputs, outputs)

# =========================
# BUILD & COMPILE
# =========================
model = hybrid_densenet_transformer()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# CALLBACKS (NO EARLY STOPPING)
# =========================
callbacks = [
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=4,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath="best_aptos_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

# =========================
# TRAINING (RUNS FULL 50 EPOCHS)
# =========================
EPOCHS = 50

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================
# ACCURACY & LOSS PLOTS
# =========================
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# =========================
# CONFUSION MATRIX
# =========================
val_data.reset()
pred_probs = model.predict(val_data)
y_pred = np.argmax(pred_probs, axis=1)
y_true = val_data.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(NUM_CLASSES),
    yticklabels=range(NUM_CLASSES)
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# =========================
# CLASSIFICATION REPORT
# =========================
print("Classification Report:")
