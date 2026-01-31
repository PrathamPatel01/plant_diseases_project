from tensorflow import keras
from tensorflow.keras import layers

def build_cnn(num_classes: int):
    model = keras.Sequential([
        layers.Input(shape=(128, 128, 3)),

        # Block 1
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        # Block 2
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        # Block 3
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        # Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),

        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

    return model
