import tensorflow as tf
import keras
from keras import layers


def build_model():
    model = keras.Sequential(
        layers=[
            keras.layers.InputLayer(shape=(4, 8)),
            keras.layers.LSTM(units=10, name="LSTM_Layer"),
            keras.layers.Dense(
                units=8, activation=keras.activations.sigmoid, name="output_Layer"
            ),
        ],
        name="LSTM_Model",
    )
    return model


def gen_compiled_LSTM_model():
    model = build_model()

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            keras.metrics.Accuracy(name="acc"),
            keras.metrics.Recall(name="Recall"),
            keras.metrics.Precision(name="Precision"),
        ],
    )

    return model
