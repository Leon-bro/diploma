import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, GlobalAveragePooling2D, Concatenate, UpSampling2D, Add, Multiply

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
    output = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(decoder)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

class HardSwish(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs * tf.nn.relu6(inputs + 3) / 6

class SqueezeExcite(tf.keras.layers.Layer):
    def __init__(self, input_channels, reduction=4):
        super(SqueezeExcite, self).__init__()
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=True)
        reduced_dim = input_channels // reduction
        self.fc1 = Conv2D(reduced_dim, 1, activation="relu")
        self.fc2 = Conv2D(input_channels, 1, activation="sigmoid")

    def call(self, inputs):
        x = self.global_avg_pool(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        return Multiply()([inputs, x])

class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, expansion_factor, stride, use_se, activation):
        super(Bottleneck, self).__init__()
        self.stride = stride
        mid_channels = in_channels * expansion_factor

        self.expand_conv = Conv2D(mid_channels, 1, padding="same", use_bias=False)
        self.expand_bn = BatchNormalization()
        self.activation = HardSwish() if activation == "hard_swish" else ReLU()

        self.depthwise_conv = DepthwiseConv2D(kernel_size, strides=stride, padding="same", use_bias=False)
        self.depthwise_bn = BatchNormalization()

        self.se = SqueezeExcite(mid_channels) if use_se else None

        self.project_conv = Conv2D(out_channels, 1, padding="same", use_bias=False)
        self.project_bn = BatchNormalization()

    def call(self, inputs):
        x = self.expand_conv(inputs)
        x = self.expand_bn(x)
        x = self.activation(x)

        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)

        if self.se:
            x = self.se(x)

        x = self.project_conv(x)
        x = self.project_bn(x)

        if self.stride == 1 and x.shape[-1] == inputs.shape[-1]:
            x = Add()([inputs, x])  # Residual connection
        return x

class MobileNetV3Small(tf.keras.Model):
    def __init__(self):
        super(MobileNetV3Small, self).__init__()

        self.initial_conv = Conv2D(16, 3, strides=2, padding="same", use_bias=False)
        self.initial_bn = BatchNormalization()
        self.initial_activation = HardSwish()

        self.bottlenecks = [
            Bottleneck(16, 16, 3, expansion_factor=1, stride=2, use_se=True, activation="relu"),
            Bottleneck(16, 24, 3, expansion_factor=4, stride=2, use_se=False, activation="relu"),
            Bottleneck(24, 24, 3, expansion_factor=3, stride=1, use_se=False, activation="relu"),
            Bottleneck(24, 40, 5, expansion_factor=3, stride=2, use_se=True, activation="hard_swish"),
            Bottleneck(40, 40, 5, expansion_factor=3, stride=1, use_se=True, activation="hard_swish"),
            Bottleneck(40, 48, 5, expansion_factor=3, stride=1, use_se=True, activation="hard_swish"),
            Bottleneck(48, 96, 5, expansion_factor=6, stride=2, use_se=True, activation="hard_swish"),
            Bottleneck(96, 96, 5, expansion_factor=6, stride=1, use_se=True, activation="hard_swish"),
            Bottleneck(96, 96, 5, expansion_factor=6, stride=1, use_se=True, activation="hard_swish"),
        ]

        self.final_conv = Conv2D(576, 1, padding="same", use_bias=False)
        self.final_bn = BatchNormalization()
        self.final_activation = HardSwish()

    def call(self, inputs):
        x = self.initial_conv(inputs)
        x = self.initial_bn(x)
        x = self.initial_activation(x)

        for bottleneck in self.bottlenecks:
            x = bottleneck(x)

        x = self.final_conv(x)
        x = self.final_bn(x)
        x = self.final_activation(x)

        return x

class DeepLabV3(tf.keras.Model):
    def __init__(self, output_channels):
        super(DeepLabV3, self).__init__()
        self.output_channels = output_channels

        # Backbone
        self.backbone = MobileNetV3Small()

        # ASPP (Atrous Spatial Pyramid Pooling) layers
        self.aspp1 = self._conv_bn_relu(256, 1, dilation_rate=1)
        self.aspp2 = self._conv_bn_relu(256, 3, dilation_rate=6)
        self.aspp3 = self._conv_bn_relu(256, 3, dilation_rate=12)
        self.aspp4 = self._conv_bn_relu(256, 3, dilation_rate=18)

        self.global_avg_pool = GlobalAveragePooling2D()
        self.global_avg_conv = self._conv_bn_relu(256, 1)

        # Concatenate and output layers
        self.concat_conv = self._conv_bn_relu(256, 1)
        self.final_conv = Conv2D(output_channels, 1, padding="same", activation="sigmoid")

    def _conv_bn_relu(self, filters, kernel_size, dilation_rate=1):
        return tf.keras.Sequential([
            Conv2D(filters, kernel_size, padding="same", dilation_rate=dilation_rate, use_bias=False),
            BatchNormalization(),
            ReLU()
        ])

    def call(self, inputs):
        # Backbone features
        x = self.backbone(inputs)

        # ASPP module
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        # Global average pooling path in ASPP
        x5 = self.global_avg_pool(x)
        x5 = tf.expand_dims(tf.expand_dims(x5, 1), 1)  # Reshape for concatenation
        x5 = self.global_avg_conv(x5)
        x5 = UpSampling2D(size=(8, 8))(x5)

        # Concatenate all ASPP branches
        x = Concatenate()([x1, x2, x3, x4, x5])
        x = self.concat_conv(x)

        # Upsampling to match input image resolution
        x = UpSampling2D(size=(4, 4), interpolation="bilinear")(x)
        return self.final_conv(x)