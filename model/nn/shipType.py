import tensorflow as tf
import keras
import numpy as np


def build_model(input_dim, output_dim):

    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(input_dim,)),
            # keras.layers.BatchNormalization(),
            keras.layers.Dense(5, activation=keras.activations.relu, name="L1"),
            # keras.layers.BatchNormalization(),
            keras.layers.Dense(
                units=output_dim,
                activation=keras.activations.sigmoid,
                name="Output_layer",
            ),
        ]
    )
    return model


def gen_compiled_ship_type_classifier_model(
    input_dim: int = 10, output_dim: int = 5
) -> keras.Model:
    # model = ClassifyShipType()
    model = build_model(input_dim=input_dim, output_dim=output_dim)

    # model.build((None, 9))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            "accuracy",
            keras.metrics.Recall(name="Recall"),
            keras.metrics.Precision(name="Precision"),
        ],
    )
    # print(model.summary(expand_nested=True))
    return model
