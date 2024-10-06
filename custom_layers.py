import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras import backend as K
class SelectiveKernelUnit(layers.Layer):
    def __init__(self, filters, reduction=8, kernel_sizes=(1, 3, 5), L=32):
        super(SelectiveKernelUnit, self).__init__()
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.reduction = reduction

        # Define the convolutions for each kernel size
        self.convs = [layers.Conv2D(filters, kernel_size=k, padding='same', use_bias=False)
                      for k in kernel_sizes]
        self.bn = layers.BatchNormalization()

        # Fully connected layers for attention mechanism with batch normalization
        self.global_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(max(filters // reduction , L), activation=None)  # First FC layer without activation
        self.bn_fc1 = layers.BatchNormalization()  # BatchNorm for the first FC layer
        self.fc2 = layers.Dense(filters * len(kernel_sizes), activation=None)  # Output for len(kernel_sizes) weights per channel

    def call(self, inputs):
        # Step 1: Multi-Scale Convolution
        feature_maps = [conv(inputs) for conv in self.convs]

        # Apply BatchNorm and ReLU to each feature map
        feature_maps = [tf.nn.relu(self.bn(fm)) for fm in feature_maps]
        # Step 2: Combine multi-scale features by adding element-wise
        U = tf.add_n(feature_maps)

        # Step 3: Global Average Pooling (channel-wise context aggregation)
        s = self.global_pool(U)  # Shape: [batch_size, channels]
        # Step 4: Fully connected layers to compute attention weights
        z = self.fc1(s)  # Reduction step, no activation yet
        z = self.bn_fc1(z)  # Apply BatchNorm after the first FC layer
        z = tf.nn.relu(z)  # ReLU activation after batch norm

        # Step 5: Compute attention weights (softmax across kernel sizes for each channel)
        z = self.fc2(z)  # Output size: [batch_size, filters * len(kernel_sizes)]
        z = tf.reshape(z, [-1, self.filters, len(self.kernel_sizes)])  # Reshape to [batch_size, filters, num_kernels]
        # Apply softmax across kernel sizes (axis 2), ensuring sum of weights across kernels for each channel equals 1
        attention_weights = tf.nn.softmax(z, axis=-1)  # Softmax across kernel sizes for each channel
        # Step 6: Apply attention weights to corresponding feature maps
        # Transpose attention_weights to [num_kernels, batch_size, filters] for easy broadcasting
        attention_weights = tf.transpose(attention_weights, [2, 0, 1])  # Shape: [num_kernels, batch_size, filters]
        # Reshape attention weights for broadcasting
        weighted_feature_maps = [fm * tf.reshape(attention_weights[i], [-1, 1, 1, self.filters])
                                 for i, fm in enumerate(feature_maps)]
        # Step 7: Fuse the weighted feature maps (element-wise sum)
        output = tf.add_n(weighted_feature_maps)

        return output

def _bernoulli(shape, mean):
    return tf.nn.relu(tf.sign(mean - tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)))


class DropBlock2D(tf.keras.layers.Layer):
    def __init__(self, keep_prob, block_size, scale=True, **kwargs):
        super(DropBlock2D, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 4
        _, self.h, self.w, self.channel = input_shape.as_list()
        #tf.print(input_shape.as_list())
        # pad the mask
        p1 = (self.block_size - 1) // 2
        p0 = (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]
        self.set_keep_prob()
        super(DropBlock2D, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(self.scale,
                             true_fn=lambda: output * tf.cast(tf.size(mask), tf.float32) / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output
        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(tf.cast(training, tf.bool)), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs,
                         false_fn=drop)
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        w, h = tf.cast(self.w, tf.float32), tf.cast(self.h, tf.float32)
        self.gamma = (1. - self.keep_prob) * (w * h) / (self.block_size ** 2) / \
                     ((w - self.block_size + 1) * (h - self.block_size + 1))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                        self.h - self.block_size + 1,
                                        self.w - self.block_size + 1,
                                        self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask

class MultiScaleResidualConvolutionModule(layers.Layer):
    """Multi-scale residual convolution module with DropBlock and spatial attention."""
    def __init__(self, filters, block_size=7, keep_prob=0.15, kernel_sizes=(1, 3, 5)):
        super(MultiScaleResidualConvolutionModule, self).__init__()
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.block_size = block_size
        self.drop_rate = keep_prob
        self.first_conv = layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')
        # DropBlock Layer
        self.drop_block = DropBlock2D(keep_prob=keep_prob, block_size=block_size)
        self.bn = layers.BatchNormalization()
        # Define the convolution layers for each kernel size
        self.convs = [layers.Conv2D(filters, kernel_size=k, padding='same', activation='relu') for k in kernel_sizes]
        self.bns = [layers.BatchNormalization() for _ in kernel_sizes]
        # Final 1x1 convolution for residual output
        self.final_conv = layers.Conv2D(filters, 1, padding='same')

    def call(self, inputs, training=None):
        x = self.first_conv(inputs)
        x = tf.nn.relu(self.bn(self.drop_block(x, training=training)))
        #tf.print("First conv shape", tf.shape(x))
        #tf.print("First conv", x)
        # Step 1: Multi-Scale Convolution
        feature_maps = [conv(inputs) for conv in self.convs]
        #tf.print("feature_maps shape", tf.shape(feature_maps))
        # Apply BatchNorm and ReLU
        feature_maps = [tf.nn.relu(bn(fm)) for fm, bn in zip(feature_maps, self.bns)]
        #tf.print("relu + bn feature_maps shape", tf.shape(feature_maps))
        # Step 2: Element-wise addition to combine multi-scale features
        combined_features = tf.add_n(feature_maps)  # Combine feature maps
        #tf.print("combined_features shape", tf.shape(combined_features))
        # Step 4: Global Average Pooling by width and height separately
        pooled_width = tf.reduce_mean(combined_features, axis=2, keepdims=True)  # Pool across width (axis 2)
        #tf.print("pooled_width shape", tf.shape(pooled_width))
        pooled_height = tf.reduce_mean(combined_features, axis=1, keepdims=True)  # Pool across height (axis 1)
        #tf.print("pooled_height shape", tf.shape(pooled_height))
        # Step 5: Multiply the pooled results to create the attention map
        attention_map = tf.sigmoid(pooled_width * pooled_height)
        #tf.print("attention_map shape", tf.shape(attention_map))
        # Step 6: Apply attention map to the dropped features
        attention_output = x * attention_map
        #tf.print("attention_output shape", tf.shape(attention_output))
        # Step 7: Residual Connection
        residual_output = self.final_conv(inputs)
        output = attention_output + residual_output  # Residual connection
        #tf.print("output shape", tf.shape(output))
        return output

class SEBlock(layers.Layer):
    def __init__(self, filters, reduction=16):
        super(SEBlock, self).__init__()
        self.filters = filters
        self.reduction = reduction
        self.gp = layers.GlobalAveragePooling2D()
        # Fully connected layers for excitation
        self.fc1 = layers.Dense(filters // reduction, activation='relu')
        self.fc2 = layers.Dense(filters, activation='sigmoid')

    def call(self, inputs):
        # Step 1: Squeeze (Global Average Pooling)
        squeeze = self.gp(inputs)  # Shape: [batch_size, channels]

        # Step 2: Excitation (Fully connected layers to get channel-wise attention weights)
        excitation = self.fc1(squeeze)  # Reduce dimensionality
        excitation = self.fc2(excitation)  # Restore original channel size
        output = tf.keras.layers.multiply([inputs, excitation])
        return output

class ResidualAttentionModule(layers.Layer):
    """Multi-scale residual convolution module with DropBlock and spatial attention."""
    def __init__(self, filters):
        super(ResidualAttentionModule, self).__init__()
        self.conv = layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')
        self.conv2 = layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')
        self.bn = layers.BatchNormalization()
        self.bn1 = layers.BatchNormalization()
        self.residual = layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')
        self.se_block = SEBlock(filters)

    def call(self, inputs, training=None):
        first = tf.nn.relu(self.bn(self.conv(inputs)))
        second = tf.nn.relu(self.bn1(self.conv2(first)))
        residual = self.residual(inputs)
        addition = second + residual
        out = self.se_block(addition)
        return out