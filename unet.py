import tensorflow as tf
from tensorflow.keras import layers, Model
from custom_layers import MultiScaleResidualConvolutionModule, SelectiveKernelUnit, ResidualAttentionModule  # Assuming your custom layers file is imported



def conv_block(inputs, filters):
    #x = MultiScaleResidualConvolutionModule(filters, keep_prob=0.15)(inputs)
    x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x


def encoder_block(inputs, filters):
    """Encoder block: Conv block followed by MaxPooling."""
    x = conv_block(inputs, filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p


def decoder_block(inputs, skip_features, filters):
    """Decoder block: Conv2DTranspose for upsampling followed by Conv block."""
    skip = SelectiveKernelUnit(filters)(skip_features)
    add = tf.keras.layers.Add()([skip_features, skip])
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, add])
    x = conv_block(x, filters)
    return x


def create_unet1(input_shape, depth, num_filters):
    """Function to create U-Net model with dynamic depth and first layer filters."""
    inputs = layers.Input(input_shape) if isinstance(input_shape, tuple) else input_shape
    encoders = []
    pool = inputs
    filters = num_filters

    # Encoder part (downsampling)
    for i in range(depth):
        encoder, pool = encoder_block(pool, filters)
        encoders.append(encoder)  # Store skip connections
        filters *= 2  # Double the filters at each step

    # Bottleneck
    bottleneck = conv_block(pool, filters)
    # Decoder part (upsampling)
    filters //= 2  # Start by halving the filters
    for i in range(depth - 1, -1, -1):
        bottleneck = decoder_block(bottleneck, encoders[i], filters)
        filters //= 2  # Halve the filters at each step

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(bottleneck)

    model = Model(inputs, outputs)
    return model