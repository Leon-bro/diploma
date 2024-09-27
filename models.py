import tensorflow as tf
from tensorflow.keras import layers, Model
from custom_layers import MultiScaleResidualConvolutionModule, SelectiveKernelUnit, ResidualAttentionModule  # Assuming your custom layers file is imported



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




class ModifiedUNet(tf.keras.Model):
    def __init__(self, num_classes, depth=4, first_filters=64, reduction=16, kernel_sizes=(1, 3, 5)):
        super(ModifiedUNet, self).__init__()
        self.num_classes = num_classes
        self.depth = depth
        self.first_filters = first_filters

        self.enc_blocks = []  # Encoder layers
        self.sku_blocks = []  # Skip connections with SKU
        self.upconv_blocks = []  # Upsampling blocks
        self.dec_blocks = []  # Decoder layers
        self.conv1x1_blocks = []  # 1x1 conv to match channels for skip connections

        # Encoder: Create the downsampling layers dynamically based on the depth
        for i in range(depth):
            filters = first_filters * (2 ** i)  # Double the filters at each level
            self.enc_blocks.append(MultiScaleResidualConvolutionModule(filters=filters, block_size=7, drop_rate=0.15, kernel_sizes=kernel_sizes))
            if i < depth - 1:
                self.sku_blocks.append(SelectiveKernelUnit(filters=filters, reduction=reduction, kernel_sizes=kernel_sizes))

        self.pool = layers.MaxPooling2D(pool_size=(2, 2), strides=2)

        # Bottleneck
        self.bottleneck = MultiScaleResidualConvolutionModule(filters=first_filters * (2 ** depth), block_size=7, drop_rate=0.15, kernel_sizes=kernel_sizes)

        # Decoder: Create the upsampling layers dynamically based on the depth
        for i in reversed(range(1, depth)):
            filters = first_filters * (2 ** i)  # Filters for decoder blocks
            self.upconv_blocks.append(layers.Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding='same'))
            self.dec_blocks.append(ResidualAttentionModule(filters=filters))
            self.conv1x1_blocks.append(layers.Conv2D(filters=filters, kernel_size=1, padding='same'))  # 1x1 conv to match channels for skip connection

        # Final 1x1 convolution to map to num_classes
        self.final_conv = layers.Conv2D(num_classes, kernel_size=1, activation='sigmoid')

    def call(self, inputs, training=None):
        # Encoder path with Multi-scale convolutions and pooling
        enc_outputs = []
        x = inputs
        for i in range(self.depth):
            x = self.enc_blocks[i](x, training=training)
            enc_outputs.append(x)
            if i < self.depth - 1:  # Apply pooling except at the bottleneck
                x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x, training=training)

        # Decoder path with upsampling and skip connections (using SKU)
        for i in range(self.depth - 1):
            x = self.upconv_blocks[i](x)  # Upsample feature map

            # Adjust skip connection using 1x1 convolution to ensure matching channels
            skip = self.conv1x1_blocks[i](enc_outputs[self.depth - 2 - i])

            # Combine the upsampled feature map with the adjusted skip connection
            x = self.sku_blocks[i](x + skip)  # Skip connection with SKU
            x = self.dec_blocks[i](x, training=training)

        # Final convolution to produce the segmentation map
        output = self.final_conv(x)

        return output