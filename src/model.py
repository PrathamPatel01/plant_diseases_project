# # from tensorflow import keras
# # from tensorflow.keras import layers

# # def build_cnn(num_classes: int):
# #     model = keras.Sequential([
# #         layers.Input(shape=(128, 128, 3)),

# #         # Block 1
# #         layers.Conv2D(32, 3, padding="same", activation="relu"),
# #         layers.MaxPooling2D(),

# #         # Block 2
# #         layers.Conv2D(64, 3, padding="same", activation="relu"),
# #         layers.MaxPooling2D(),

# #         # Block 3
# #         layers.Conv2D(128, 3, padding="same", activation="relu"),
# #         layers.MaxPooling2D(),

# #         # Head
# #         layers.GlobalAveragePooling2D(),
# #         layers.Dense(128, activation="relu"),

# #         layers.Dropout(0.3),
# #         layers.Dense(num_classes, activation="softmax"),
# #     ])

# #     model.compile(
# #     optimizer="adam",
# #     loss="sparse_categorical_crossentropy",
# #     metrics=["accuracy"]
# # )

# #     return model


# # src/model.py
# # src/model.py
# # src/model.py
# # src/model.py
# from tensorflow import keras
# from tensorflow.keras import layers

# def build_cnn(num_classes: int):
#     model = keras.Sequential([
#         layers.Input(shape=(128, 128, 3)),

#         # Block 1
#         layers.Conv2D(32, 3, padding="same", activation="relu"),
#         layers.MaxPooling2D(),

#         # Block 2
#         layers.Conv2D(64, 3, padding="same", activation="relu"),
#         layers.MaxPooling2D(),

#         # Block 3
#         layers.Conv2D(128, 3, padding="same", activation="relu"),
#         layers.MaxPooling2D(),

#         # Head
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(128, activation="relu"),

#         layers.Dropout(0.3),
#         layers.Dense(num_classes, activation="softmax"),
#     ])

#     model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

#     return model

from src.config import IMAGE_SIZE, CHANNELS, DROPOUT_RATE, LEARNING_RATE


def build_model(num_classes: int, train_base: bool = False):
    """
    Build a transfer learning model using MobileNetV2.

    Why MobileNetV2?
    - Better than a small custom CNN for image classification
    - Lightweight
    - Fast enough for local training/demo
    - Good choice for portfolio-level ML projects

    Args:
        num_classes: Number of plant disease classes.
        train_base: Whether to fine-tune the pretrained MobileNetV2 base.

    Returns:
        Compiled Keras model.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    input_shape = (*IMAGE_SIZE, CHANNELS)

    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = train_base

    inputs = keras.Input(shape=input_shape)

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="classification_head",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="plant_disease_mobilenetv2",
    )

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def print_model_summary(num_classes: int):
    """
    Utility function to quickly inspect the model.
    """
    model = build_model(num_classes=num_classes)
    model.summary()