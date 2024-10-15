import tensorflow as tf
from tensorflow.keras import layers, Model
import custom_layers


def conv_block(inputs, filters):
    """Basic convolutional block with two Conv2D layers."""
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
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, filters)
    return x


def create_unet(input_shape, depth, num_filters):
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


def create_w_net(input_shape, depth, num_filters):
    """Creates a W-Net model by stacking two U-Nets."""
    # First U-Net
    inputs = layers.Input(shape=input_shape)
    unet1 = create_unet(inputs, depth, num_filters)

    # Get the output from the first U-Net
    output_unet1 = unet1(inputs)

    # Concatenate the input and the output of the first U-Net
    concatenated_input = layers.Concatenate()([inputs, output_unet1])

    # Second U-Net with concatenated input
    unet2 = create_unet(concatenated_input, depth, num_filters)

    # Output from the second U-Net
    output_unet2 = unet2(concatenated_input)

    # Create the final W-Net model
    final_model = Model(inputs=inputs, outputs=[output_unet1, output_unet2])

    return final_model




def modified_encoder_block(inputs, filters, keep_prob=0.85, block_size=3):
    """Encoder block: Multi-scale residual block followed by MaxPooling."""
    x = custom_layers.MultiScaleResidualConvolutionModule(filters, keep_prob=keep_prob, block_size=block_size)(inputs)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def modified_skip_connection_block(inputs, filters, reduction, L):
    """Skip connection block using Selective Kernel Unit (SKU)."""
    sku = custom_layers.SelectiveKernelUnit(filters, reduction=reduction, L=L)(inputs)
    add = layers.Add()([inputs, sku])
    return add

def modified_decoder_block(inputs, skip_features, filters):
    """Decoder block: Conv2DTranspose for upsampling followed by Conv block."""
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = custom_layers.ResidualAttentionModule(filters)(x)
    return x

def build_munet(input_shape=(128, 128, 1), first_filters=32, depth=4, keep_prob=0.85, block_size=3, reduction=8, L=16):
    """Build the Modified U-Net with parametrized depth and first_filters."""
    inputs = layers.Input(shape=input_shape)

    # Encoder path
    encoders = []
    pools = []
    skips = []  # Store skip connections
    filters = first_filters

    for i in range(depth):
        x, p = modified_encoder_block(inputs if i == 0 else pools[-1], filters, keep_prob=keep_prob, block_size=block_size)
        encoders.append(x)  # Store encoder outputs
        pools.append(p)  # Store pooled outputs for next depth level

        # Apply skip connection (SKU) to encoder output
        s = modified_skip_connection_block(x, filters, reduction, L)
        skips.append(s)  # Store skip connections for decoder
        filters *= 2  # Double the filters at each depth level

    # Bottleneck
    bottleneck = modified_skip_connection_block(pools[-1], filters // 2, reduction, L)  # Keep bottleneck with same filters as last encoder

    # Decoder path
    filters //= 2  # Reduce filters at each decoding step
    decoder = bottleneck

    for i in range(depth - 1, -1, -1):
        decoder = modified_decoder_block(decoder, skips[i], filters)  # Use skip connections from the encoder
        filters //= 2  # Halve the filters at each depth level

    # Output layer
    output = layers.Conv2D(1, (1, 1), padding="same")(decoder)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model