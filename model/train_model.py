import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ───────────────── CONFIG ─────────────────
TRAIN_DIR    = "dataset/train"
VAL_DIR      = "dataset/test"
MODEL_SAVE   = "model/skin_disease_model.h5"
CLASSES_SAVE = "model/class_names.txt"

IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 20
LEARNING_RATE = 1e-4


# ───────────────── MODEL ─────────────────
def build_model(num_classes: int):
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


# ───────────────── TRAIN ─────────────────
def train():
    print("=" * 60)
    print("Skin Disease Detection — Training Started")
    print("=" * 60)

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    print(f"\n📂 Loading training data from: {TRAIN_DIR}")
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    print(f"\n📂 Loading validation data from: {VAL_DIR}")
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    # Class info
    class_names = list(train_gen.class_indices.keys())
    num_classes = len(class_names)

    print(f"\n✅ Classes detected: {num_classes}")
    print("Class mapping:", train_gen.class_indices)

    # Build model
    model, base_model = build_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE, monitor="val_accuracy", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    ]

    # ─── Phase 1: Train head ───
    print("\n🚀 Phase 1: Training classifier head...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # ─── Phase 2: Fine-tuning ───
    print("\n🔓 Phase 2: Fine-tuning base model...")
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=callbacks
    )

    # Save class names
    os.makedirs("model", exist_ok=True)
    with open(CLASSES_SAVE, "w") as f:
        for name in class_names:
            f.write(name + "\n")

    print(f"\n✅ Model saved at: {MODEL_SAVE}")
    print(f"✅ Class names saved at: {CLASSES_SAVE}")

    # Final evaluation
    loss, acc = model.evaluate(val_gen, verbose=0)
    print(f"\n📊 Final Accuracy: {acc * 100:.2f}%")
    print(f"📊 Final Loss: {loss:.4f}")


if __name__ == "__main__":
    train()