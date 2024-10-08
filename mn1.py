import tensorflow as tf
from tensorflow.keras import layers
import custom_layers  # Assuming your custom layers are imported
import models

# Example usage
model = models.build_munet(input_shape=(128, 128, 1), first_filters=32, depth=4)
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='munet.png')
