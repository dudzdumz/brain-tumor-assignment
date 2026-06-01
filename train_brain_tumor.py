import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# -----------------------------
# Step 1: Settings
# -----------------------------
DATASET_DIR = "dataset"
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10


# -----------------------------
# Step 2: Load images
# -----------------------------
def load_dataset():
    images = []
    labels = []

    class_map = {
        "no": 0,
        "yes": 1
    }

    for class_name, label in class_map.items():
        folder_path = os.path.join(DATASET_DIR, class_name)

        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)

            try:
                img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
                img_array = img_to_array(img) / 255.0

                images.append(img_array)
                labels.append(label)

            except Exception as e:
                print(f"Could not load image {image_path}: {e}")

    return np.array(images), np.array(labels)


X, y = load_dataset()

print("Dataset loaded successfully.")
print("Total images:", len(X))
print("Image shape:", X[0].shape)
print("Class labels: 0 = No Tumor, 1 = Tumor")
print("No Tumor images:", np.sum(y == 0))
print("Tumor images:", np.sum(y == 1))


# -----------------------------
# Step 3: Train / test split
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("\nData split completed.")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))


# -----------------------------
# Step 4: Data augmentation
# -----------------------------
train_datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow(
    X_train, y_train, batch_size=BATCH_SIZE
)


# -----------------------------
# Step 5: Custom CNN model
# -----------------------------
def build_custom_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


custom_cnn = build_custom_cnn()
custom_cnn.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\nTraining Custom CNN...")
history_cnn = custom_cnn.fit(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=[early_stop]
)


# -----------------------------
# Step 6: MobileNetV2 model
# -----------------------------
def build_mobilenet_model():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


mobilenet_model = build_mobilenet_model()
mobilenet_model.summary()

print("\nTraining MobileNetV2 Transfer Learning Model...")
history_mobilenet = mobilenet_model.fit(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=[early_stop]
)


# -----------------------------
# Step 7: Evaluation function
# -----------------------------
def evaluate_model(model, model_name):
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int32").flatten()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print(f"RESULTS: {model_name}")
    print("=" * 50)
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Tumor", "Tumor"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return accuracy, precision, recall, f1


cnn_results = evaluate_model(custom_cnn, "Custom CNN")
mobilenet_results = evaluate_model(mobilenet_model, "MobileNetV2 Transfer Learning")


# -----------------------------
# Step 8: Plot training graphs
# -----------------------------
os.makedirs("outputs", exist_ok=True)

def plot_history(history, model_name, filename):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join("outputs", filename))
    plt.show()


plot_history(history_cnn, "Custom CNN", "custom_cnn_training_graph.png")
plot_history(history_mobilenet, "MobileNetV2", "mobilenet_training_graph.png")


# -----------------------------
# Step 9: Final comparison table
# -----------------------------
print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)
print(f"{'Metric':<15}{'Custom CNN':<20}{'MobileNetV2':<20}")
print("-" * 70)
print(f"{'Accuracy':<15}{cnn_results[0] * 100:<20.2f}{mobilenet_results[0] * 100:<20.2f}")
print(f"{'Precision':<15}{cnn_results[1] * 100:<20.2f}{mobilenet_results[1] * 100:<20.2f}")
print(f"{'Recall':<15}{cnn_results[2] * 100:<20.2f}{mobilenet_results[2] * 100:<20.2f}")
print(f"{'F1 Score':<15}{cnn_results[3] * 100:<20.2f}{mobilenet_results[3] * 100:<20.2f}")