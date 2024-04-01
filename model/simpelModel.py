import tensorflow as tf


class SimpleModel(tf.keras.Model):
    INPUTS = 19
    TARGETS = 6

    def __init__(self, input_dim = 9 ,output_bias=None):
        super(SimpleModel, self).__init__(name='SimpleModel')
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)

        self.batchNorm = tf.keras.layers.BatchNormalization(name='BatchNormalization')
        self.Layer1 = tf.keras.layers.Dense(10, activation=tf.nn.relu, input_dim=input_dim,
                                            name='dense1')  # input shape required
        self.Layer2 = tf.keras.layers.Dense(30, activation=tf.nn.relu, name='dense2')
        self.Layer3 = tf.keras.layers.Dense(self.TARGETS, activation=tf.nn.softmax, name='dense3',
                                            bias_initializer=output_bias)
        # x = tf.random.normal(size=(1, 32, 32, 3))
        # x = tf.convert_to_tensor(x)
        # _ = self.call(x)

    def call(self, inputs, training=None, mask=None):
        x = self.batchNorm(inputs)
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        return x
