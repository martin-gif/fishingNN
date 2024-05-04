import tensorflow as tf
import keras
from keras import layers


def build_model():
    model = keras.Sequential(
        layers=[
            keras.layers.InputLayer(shape=(4, 8)),
            keras.layers.LSTM(units=5, name="LSTM_Layer"),
            keras.layers.Dense(
                units=8, activation=keras.activations.sigmoid, name="output_Layer"
            ),
            keras.layers.Reshape(target_shape=(1, 8)),
        ],
        name="LSTM_Model",
    )
    return model


def gen_compiled_LSTM_model():
    model = build_model()

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError(),
        metrics=[
            keras.metrics.CosineSimilarity(name="cosine"),
            keras.metrics.Recall(name="Recall"),
            keras.metrics.Precision(name="Precision"),
        ],
    )

    return model
