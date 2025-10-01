import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

IMG_SIZE = 224  
DATA_DIR = r"Dataset" 
MODEL_PATH = "busi_cnn_model.h5"


def load_data():
    X, y = [], []
    print(f"Loading dataset from: {DATA_DIR}")

    for label in ["cancer", "normal"]:   # match folder names
        path = os.path.join(DATA_DIR, label)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder not found: {path}")

        count = 0
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Skipping unreadable file: {img_path}")
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(1 if label == "cancer" else 0)
            count += 1
        print(f"Loaded {count} images for class '{label}'")

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # normalize
    y = to_categorical(y, num_classes=2)

    print(f"Total samples: {len(X)} (Cancer={np.sum(np.argmax(y,axis=1)==1)}, Normal={np.sum(np.argmax(y,axis=1)==0)})")
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_model():
    print("Building CNN model...")
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax")  
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def train_model():
    X_train, X_val, y_train, y_val = load_data()
    model = build_model()
    print("Starting training...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)
    model.save(MODEL_PATH)
    print(f"Model trained & saved as '{MODEL_PATH}'")


if __name__ == "__main__":
    train_model()
