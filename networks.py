import os
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, LeakyReLU, Flatten, MaxPooling2D, Input



def model_maker(target_size, model_id):
    """ This function creates a trainable model. 
        params:
            target_size: tuple, size of the input image to the network
            model_id: integer, it can be 1 to 4
        returns: 
            tensorflow trainable model.
    """

    if model_id == 1:
        inp = Input(shape = (*target_size, 3), name = 'input_layer')
        cnv1 = Conv2D(filters = 10, kernel_size = (3, 3), strides = (1, 1), name = 'conv_1')(inp)
        cnv2 = Conv2D(filters = 10, kernel_size = (3, 3), strides = (1, 1), name = 'conv_2')(cnv1)
        mxp1 = MaxPooling2D(pool_size = (2, 2), strides= (2, 2), name = 'maxpool_1')(cnv2)
        cnv3 = Conv2D(filters = 16, kernel_size = (3, 3), strides = (1, 1), name = 'conv_3')(mxp1)
        cnv4 = Conv2D(filters = 16, kernel_size = (3, 3), strides = (1, 1), name = 'conv_4')(cnv3)
        mxp2 = MaxPooling2D(pool_size = (2, 2), strides= (2, 2), name = 'maxpool_2')(cnv4)
        fltn = Flatten(name = 'flatten_layer')(mxp2)
        FC1 = Dense(50, name = 'FC_1')(fltn)
        FC1 = LeakyReLU(alpha = 0.3, name = 'leaky_ReLu_1')(FC1)
        FC2 = Dense(50, name = 'FC_2')(FC1)
        FC2 = LeakyReLU(alpha = 0.3, name = 'leaky_ReLu_2')(FC2)
        output = Dense(3, activation = 'softmax', name = 'output_layer')(FC2)
        model = Model(inputs = inp, outputs = output, name = 'WheatClassifier_CNN_'+str(model_id))
        model.summary()

    return model
