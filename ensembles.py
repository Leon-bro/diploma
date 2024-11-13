import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, GlobalAveragePooling2D, Concatenate, UpSampling2D, Add, Multiply

import custom_layers

def build_reverse_unet(input_, filters=8):
    inputs = input_
    
    # Encoder
    # 8 
    conv1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv1)
    up1 = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(conv1)
    filters *= 2 # 16
    conv2 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(up1)
    conv2 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv2)
    up2 = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(conv2)
    filters *= 2 #32
    # Bottleneck
    conv3 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(up2)
    conv3 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv3)

    # Decoder
    down1 = layers.MaxPooling2D(pool_size=(2,2))(conv3)
    down1 = layers.concatenate([down1, conv2], axis=3)
    filters //= 2 # 16
    conv4 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(down1)
    conv4 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv4)

    down2 = layers.MaxPooling2D(pool_size=(2,2))(conv4)
    down2 = layers.concatenate([down2, conv1], axis=3)
    filters //= 2 # 8
    conv5 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(down2)
    conv5 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv5)


    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    return Model(inputs=[inputs], outputs=[outputs])

def build_standart_unet(input_, filters=8):
    inputs = input_

    # Encoder
    # 8
    conv1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    filters *= 2 # 16
    
    conv2 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    filters *= 2 #32
    conv3 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv3)

    # Decoder
    filters //= 2 # 16
    up6 = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(conv3)
    up6 = layers.concatenate([up6, conv2], axis=3)
    conv4 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(up6)
    conv4 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv4)

    filters //= 2 # 8
    
    up7 = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(conv4)
    up7 = layers.concatenate([up7, conv1], axis=3)
    conv5 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(up7)
    conv5 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv5)


    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    return Model(inputs=[inputs], outputs=[outputs])

def build_reversed_filters_unet(input_, filters=32):
    inputs = input_

    # Encoder
    # 32
    conv1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    filters //= 2 # 16

    conv2 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    filters //= 2 # 8
    conv3 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv3)

    # Decoder
    filters *= 2 # 16
    up6 = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(conv3)
    up6 = layers.concatenate([up6, conv2], axis=3)
    conv4 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(up6)
    conv4 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv4)

    filters *= 2 # 32

    up7 = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(conv4)
    up7 = layers.concatenate([up7, conv1], axis=3)
    conv5 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(up7)
    conv5 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv5)


    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    return Model(inputs=[inputs], outputs=[outputs])

def create_w_net(input_, unet_build):
    """Creates a W-Net model by stacking two U-Nets."""
    # First U-Net
    inputs = input_
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
    w_net1 = create_w_net(inputs, build_reverse_unet)
    w_net2 = create_w_net(inputs, build_standart_unet)
    w_net3 = create_w_net(inputs, build_reversed_filters_unet)
    concatenated_output = layers.Concatenate()([w_net1(inputs)[-1], w_net2(inputs)[-1],  w_net3(inputs)[-1]])
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(concatenated_output)
    return Model(inputs=inputs, outputs=outputs)

model = create_w_net_ensemble((256,256,1))
model.summary()