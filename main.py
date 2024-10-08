import tensorflow as tf
from tensorflow.keras import layers
import custom_layers

def encoder_block(inputs, filters):
    """Encoder block: Conv block followed by MaxPooling."""
    x = custom_layers.MultiScaleResidualConvolutionModule(filters)(inputs)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def skip_connection_block(inputs, filters):
    sku = custom_layers.SelectiveKernelUnit(filters)(inputs)
    add = layers.Add()([inputs, sku])
    return add

def decoder_block(inputs, skip_features, filters):
    """Decoder block: Conv2DTranspose for upsampling followed by Conv block."""
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = custom_layers.ResidualAttentionModule(filters)(x)
    return x

def MUnet():
    inputs = layers.Input(shape=(128, 128, 1))
    
    #encoders
    m1, p1 = encoder_block(inputs, 32)
    m2, p2 = encoder_block(p1, 64)
    m3, p3 = encoder_block(p2, 128)
    m4, p4 = encoder_block(p3, 256)
    
    #skip_connections
    s1 = skip_connection_block(m1, 32)
    s2 = skip_connection_block(m2, 64)
    s3 = skip_connection_block(m3, 128)
    s4 = skip_connection_block(m4, 256)
    
    #bottleneck
    s5 = skip_connection_block(p4, 256)
    
    #decoder
    
    d1 = decoder_block(s5, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)
    
    #output
    output = layers.Conv2D(1, (1, 1), padding="same")(d4)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

model = MUnet()
model.summary()

tf.keras.utils.plot_model(model, show_shapes=True, to_file='model.png')