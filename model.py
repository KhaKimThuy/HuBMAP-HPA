import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from pathlib import Path
import numpy as np
import math
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate

def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    conv = Conv2D(n_filters, # Number of filters
                  3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters, # Number of filters
                  3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)
    if max_pooling:
        next_layer = MaxPooling2D((2,2))(conv)
    else:
        next_layer = conv
    skip_connection = conv
    return next_layer, skip_connection

def upsampling_block(expansive_input, contractive_input, n_filters=32):
    up = Conv2DTranspose(
                 n_filters,    # number of filters
                 (3,3),    # Kernel size
                 strides=(2,2),
                 padding='same')(expansive_input)
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,   # Number of filters
                 (3,3),     # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters,  # Number of filters
                 (3,3),   # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv)
    return conv

def unet_model(input_size=(512,512, 3), n_filters=32, n_classes=1):
    inputs = Input(input_size)
    cblock1 = conv_block(inputs,n_filters)
    cblock2 = conv_block(cblock1[0], 2*n_filters)
    cblock3 = conv_block(cblock2[0],4*n_filters)
    cblock4 = conv_block(cblock3[0],8*n_filters) # Include a dropout_prob of 0.3 for this layer
    cblock5 = conv_block(cblock4[0],16*n_filters, dropout_prob=0.3)
    cblock6 = conv_block(cblock5[0],32*n_filters,dropout_prob=0.3,max_pooling=False)
    ublock7 = upsampling_block(cblock6[0],cblock5[1],16*n_filters)
    ublock8 = upsampling_block(ublock7,cblock4[1],  8*n_filters)
    ublock9 = upsampling_block(ublock8, cblock3[1],  4*n_filters)
    ublock10 = upsampling_block(ublock9, cblock2[1],  2*n_filters)
    ublock11 = upsampling_block(ublock10, cblock1[1],  n_filters)
    conv12 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(ublock11)
    conv13 = Conv2D(n_classes, 1, padding='same', activation='sigmoid')(conv12)
    model = tf.keras.Model(inputs=inputs, outputs=conv13)
    return model