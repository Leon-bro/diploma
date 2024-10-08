import tensorflow as tf
from tensorflow.keras import layers, Model
import custom_layers
# Define the standard convolutional block
def conv_block(input_tensor, num_filters, use_msrcm=True, block_size=7, keep_prob=0.85, kernel_sizes=(1,3,5)):
    if use_msrcm:
        x = custom_layers.MultiScaleResidualConvolutionModule(num_filters, block_size, keep_prob, kernel_sizes)
    else:
        x = layers.Conv2D(num_filters, 3, padding="same")(input_tensor)
        x = layers.ReLU()(x)
        x = layers.Conv2D(num_filters, 3, padding="same")(x)
        x = layers.ReLU()(x)
    return x

# Define the skip connection function
def skip_connect(encoder_output, decoder_input, use_ram=True, filters=4 ):
    if use_ram:
        custom_layers.SelectiveKernelUnit(filters=filters)(encoder_output)
        
    return layers.Concatenate()([encoder_output, decoder_input])


def UNet(input_shape=(128, 128, 1), depth=4, initial_filters=64):
    inputs = layers.Input(input_shape)

    # Lists to store encoder layers for skip connections
    encoder_outputs = []
    x = inputs

    # Contracting Path (Encoder)
    filters = initial_filters
    for d in range(depth):
        x = conv_block(x, filters)
        encoder_outputs.append(x)
        x = layers.MaxPooling2D((2, 2))(x)
        filters *= 2  # Double the number of filters at each depth level

    # Bottleneck
    x = conv_block(x, filters)

    # Expansive Path (Decoder)
    for d in reversed(range(depth)):
        filters //= 2  # Halve the number of filters at each decoding step
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(x)
        x = skip_connect(encoder_outputs[d], x)
        x = conv_block(x, filters)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(x)

    # Create the U-Net model
    model = Model(inputs, outputs)

    return model