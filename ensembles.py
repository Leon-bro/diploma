import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, GlobalAveragePooling2D, Concatenate, UpSampling2D, Add, Multiply
import numpy as np
import custom_layers
import models

def build_reversed_filters_unet(input_, filters=8):
    inputs = input_

    # Encoder
    # 32
    conv1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    filters //= 2  # 16
    conv2 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    filters //= 2  # 8
    conv3 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)

    # Decoder
    filters *= 2  # 16
    up6 = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(conv3)
    up6 = layers.concatenate([up6, conv2], axis=3)
    conv4 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(up6)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = layers.BatchNormalization()(conv4)

    filters *= 2  # 32
    up7 = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(conv4)
    up7 = layers.concatenate([up7, conv1], axis=3)
    conv5 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(up7)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = layers.BatchNormalization()(conv5)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    return Model(inputs=[inputs], outputs=[outputs])
def build_standart_unet(input_, filters=8):
    model = models.create_unet(input_, 2,8)

    return model

def build_reverse_unet(input_, filters=8):
    inputs = input_

    # Encoder
    # 8
    conv1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    up1 = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(conv1)

    filters *= 2  # 16
    conv2 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(up1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    up2 = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(conv2)

    filters *= 2  # 32
    # Bottleneck
    conv3 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(up2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)

    # Decoder
    down1 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    down1 = layers.concatenate([down1, conv2], axis=3)
    filters //= 2  # 16
    conv4 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(down1)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = layers.BatchNormalization()(conv4)

    down2 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    down2 = layers.concatenate([down2, conv1], axis=3)
    filters //= 2  # 8
    conv5 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(down2)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = layers.BatchNormalization()(conv5)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    return Model(inputs=[inputs], outputs=[outputs])

def create_w_net(input_, unet_build):
    """Creates a W-Net model by stacking two U-Nets."""
    # First U-Net
    inputs = layers.Input(input_) if isinstance(input_, tuple) else input_
    unet1 = unet_build(inputs)

    # Get the output from the first U-Net
    output_unet1 = unet1(inputs)

    # Concatenate the input and the output of the first U-Net
    concatenated_input = layers.Concatenate()([inputs, output_unet1])

    # Second U-Net with concatenated input
    unet2 = unet_build(concatenated_input)

    # Output from the second U-Net
    output_unet2 = unet2(concatenated_input)

    # Create the final W-Net model
    final_model = Model(inputs=inputs, outputs=[output_unet1, output_unet2])

    return final_model


def create_w_net_ensemble(input_shape):
    inputs = layers.Input(shape=input_shape)
    # Load each model with the given weights
    munet = models.build_munet((256, 256, 1), first_filters=32, depth=4, keep_prob=0.85, block_size=7, reduction=8, L=16)
    munet.load_weights("ensemble/munet_with_amplification256x256d4f32b7kp0.85.h5")

    unet = models.create_unet((256, 256, 1), depth=4, num_filters=32)
    unet.load_weights("ensemble/unet_with_amplification256x256d4f32.h5")

    wnet = models.create_w_net((256, 256, 1), depth=3, num_filters=32)
    wnet.load_weights("ensemble/wnet_with_amplification256x256d3f32.h5")

    # Obtain only the last output from each model
    output1 = munet(inputs)
    output2 = unet(inputs)
    output3 = wnet(inputs)[-1]

    # Binarize outputs based on threshold of 0.5
    binary_output1 = tf.cast(output1 > 0.5, tf.int32)
    binary_output2 = tf.cast(output2 > 0.5, tf.int32)
    binary_output3 = tf.cast(output3 > 0.5, tf.int32)

    # Apply majority voting
    votes = binary_output1 + binary_output2 + binary_output3
    final_output = tf.cast(votes >= 2, tf.float32)  # If at least 2 models voted 1, result is 1; else 0

    return Model(inputs=inputs, outputs=final_output)
    """output = tf.reduce_mean([output1, output2, output3], axis=0)
    binary_output = tf.cast(output > 0.5, tf.int32)
    return Model(inputs=inputs, outputs=final_output)"""
