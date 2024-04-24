import tensorflow as tf
import keras
import numpy as np


class ClassifyShipType(keras.Model):
    INPUTS = 19
    TARGETS = 6

    def __init__(self, input_dim=9, output_bias=None):
        super(ClassifyShipType, self).__init__(name="SimpleModel")
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)
        # Define Input layer
        # self.input_layer = keras.layers.Input(shape=(9,))

        # Layer 1
        self.batchNorm = keras.layers.BatchNormalization(name="BatchNormalization")
        self.Layer1 = keras.layers.Dense(
            10,
            activation=tf.nn.relu,
            # input_dim=input_dim,
            name="dense1",
        )  # input shape required

        # Layer 2
        self.Layer2 = keras.layers.Dense(10, activation=tf.nn.relu, name="dense2")

        # Layer 3
        self.Layer3 = keras.layers.Dense(
            self.TARGETS,
            activation=tf.nn.softmax,
            name="dense3",
            bias_initializer=output_bias,
        )

        # self.call(self.input_layer)
        # x = tf.random.normal(size=(1, 32, 32, 3))
        # x = tf.convert_to_tensor(x)
        # _ = self.call(x)

    def call(self, inputs, training=None, mask=None):
        x = self.batchNorm(inputs)
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        return x

    def model(self):
        x = keras.Input(shape=(9,))
        return keras.Model(inputs=x, outputs=self.call(x))


def build_model():

    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(8,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(10, activation=keras.activations.softmax, name="L1"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(
                7, activation=keras.activations.sigmoid, name="Output_layer"
            ),
        ]
    )
    return model


def gen_compiled_ship_type_classifier_model() -> keras.Model:
    # model = ClassifyShipType()
    model = build_model()

    # model.build((None, 9))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            "accuracy",
            keras.metrics.Recall(name="Recall"),
            keras.metrics.Precision(name="Precision"),
        ],
    )
    # print(model.summary(expand_nested=True))
    return model


if __name__ == "__main__":
    model = gen_compiled_ship_type_classifier_model()
    x = np.random.random((50, 9))
    y = np.random.randint(low=0, high=6, size=50)
    y = np.zeros((y.size, y.max() + 1))  # turn into one hot vector
    print("datapoint", x[0])
    print("lable", y[0])
    model.fit(x=x, y=y, batch_size=1, epochs=1)
    print("predict", model.predict([x[0]]))
