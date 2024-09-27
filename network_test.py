import tensorflow as tf
from tensorflow.keras import layers
from custom_layers import MultiScaleResidualConvolutionModule, SelectiveKernelUnit, ResidualAttentionModule  # Assuming your custom layers file is imported


class ModifiedUNet(tf.keras.Model):
    def __init__(self, input_shape, num_classes, depth=4, first_filters=64, reduction=16, kernel_sizes=(1, 3, 5)):
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
            self.enc_blocks.append(MultiScaleResidualConvolutionModule(filters=filters, block_size=7, keep_prob=0.15, kernel_sizes=kernel_sizes))
            if i < depth - 1:
                self.sku_blocks.append(SelectiveKernelUnit(filters=filters, reduction=reduction, kernel_sizes=kernel_sizes))

        self.pool = layers.MaxPooling2D(pool_size=(2, 2), strides=2)

        # Bottleneck
        self.bottleneck = MultiScaleResidualConvolutionModule(filters=first_filters * (2 ** depth), block_size=7, keep_prob=0.15, kernel_sizes=kernel_sizes)

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

# Assuming your model is already instantiated
input_shape = (24, 24, 1)
num_classes = 1
model = ModifiedUNet(input_shape, num_classes, depth=4, first_filters=64)

# Build the model
model.build((None, *input_shape))

# Create a dummy input and target output
dummy_input = tf.random.normal((1, *input_shape))  # Single random image (batch_size=1)
dummy_target = tf.random.normal((1, 24, 24, num_classes))  # Single random target (e.g., segmentation mask)

# Define a simple loss function for testing
loss_fn = tf.keras.losses.MeanSquaredError()

# Use tf.GradientTape to check the gradients
with tf.GradientTape() as tape:
    tape.watch(model.trainable_variables)  # Watch all trainable variables in the model
    predictions = model(dummy_input)       # Forward pass
    loss = loss_fn(dummy_target, predictions)  # Compute a dummy loss

# Compute the gradients with respect to the loss
gradients = tape.gradient(loss, model.trainable_variables)

# Check the gradients for each layer
for var, grad in zip(model.trainable_variables, gradients):
    if grad is None:
        print(f"WARNING: No gradient for {var.name}")
    else:
        print(f"Gradient computed for {var.name}, mean value: {tf.reduce_mean(grad).numpy()}")