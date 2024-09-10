import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(inputs, filters):
    """Basic convolutional block with two Conv2D layers."""
    x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    return x

def encoder_block(inputs, filters):
    """Encoder block: Conv block followed by MaxPooling."""
    x = conv_block(inputs, filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, filters):
    """Decoder block: Conv2DTranspose for upsampling followed by Conv block."""
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, filters)
    return x

def build_unet(input_shape):
    inputs = tf.keras.layers.Input((512, 512, 3))
    # Encoder
    s1, p1 = encoder_block(inputs, 8)    # 1st layer: 8 filters
    s2, p2 = encoder_block(p1, 16)       # 2nd layer: 16 filters

    # Bottleneck
    b1 = conv_block(p2, 32)              # Bottleneck with 64 filters

    # Decoder
    d2 = decoder_block(b1, s2, 16)       # 16 filters
    d3 = decoder_block(d2, s1, 8)        # 8 filters

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", name="out")(d3)

    model = Model(inputs, outputs)
    return model

def build_w_net(input_shape):
    inputs = layers.Input(input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 8)    # 1st layer: 8 filters
    s2, p2 = encoder_block(p1, 16)       # 2nd layer: 16 filters

    # Bottleneck
    b1 = conv_block(p2, 32)              # Bottleneck with 64 filters

    # Decoder
    d2 = decoder_block(b1, s2, 16)       # 16 filters
    d3 = decoder_block(d2, s1, 8)        # 8 filters


    # Output layer
    outputs1 = layers.Conv2D(1, (1, 1), activation="sigmoid", name="out1")(d3)
    x = layers.Concatenate()([inputs, outputs1])
    # Encoder
    s1, p1 = encoder_block(x, 8)    # 1st layer: 8 filters
    s2, p2 = encoder_block(p1, 16)       # 2nd layer: 16 filters

    # Bottleneck
    b1 = conv_block(p2, 32)              # Bottleneck with 64 filters

    # Decoder
    d2 = decoder_block(b1, s2, 16)       # 16 filters
    d3 = decoder_block(d2, s1, 8)        # 8 filters

    outputs2 = layers.Conv2D(1, (1, 1), activation="sigmoid", name="out2")(d3)
    model = Model(inputs, [outputs1, outputs2])
    return model