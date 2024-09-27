import tensorflow as tf
from tensorflow.keras import layers
import custom_layers


# Create a simple model with your SKU layer
#inputs = tf.keras.Input(shape=(5, 5, 1))
#outputs = SelectiveKernelUnit(filters=64)(inputs)
#model = tf.keras.Model(inputs, outputs)

# Dummy inputs and target
data = tf.random.normal((1, 6, 6, 1))
y_true = tf.random.normal((1, 5, 5, 1))

inputs = tf.keras.Input(shape=(6, 6, 1))
outputs = ResidualAttentionModule(64)(inputs)
model = tf.keras.Model(inputs, outputs)
model.summary()
model(data)