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


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def create_vit_classifier(input_shape, patch_size, num_patches, projection_dim, transformer_units, transformer_layers, model_name, num_heads, mlp_head_units, num_classes):
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    #representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes, activation = 'softmax')(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits, name = model_name)
    return model

def model_maker(target_size, model_id, num_classes = 3):
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
        output = Dense(num_classes, activation = 'softmax', name = 'output_layer')(FC2)
        model = Model(inputs = inp, outputs = output, name = 'WheatClassifier_CNN_'+str(model_id))
        model.summary()
    
    if model_id == 2:
        inp = Input(shape = (*target_size, 3), name = 'input_layer')
        cnv1 = Conv2D(filters = 10, kernel_size = (3, 3), strides = (1, 1), name = 'conv_1')(inp)
        cnv2 = Conv2D(filters = 10, kernel_size = (3, 3), strides = (1, 1), name = 'conv_2')(cnv1)
        mxp1 = MaxPooling2D(pool_size = (2, 2), strides= (2, 2), name = 'maxpool_1')(cnv2)
        cnv3 = Conv2D(filters = 16, kernel_size = (3, 3), strides = (1, 1), name = 'conv_3')(mxp1)
        cnv4 = Conv2D(filters = 16, kernel_size = (3, 3), strides = (1, 1), name = 'conv_4')(cnv3)
        mxp2 = MaxPooling2D(pool_size = (2, 2), strides= (2, 2), name = 'maxpool_2')(cnv4)
        cnv5 = Conv2D(filters = 16, kernel_size = (3, 3), strides = (1, 1), name = 'conv_3')(mxp2)
        cnv6 = Conv2D(filters = 16, kernel_size = (3, 3), strides = (1, 1), name = 'conv_4')(cnv5)
        mxp3 = MaxPooling2D(pool_size = (2, 2), strides= (2, 2), name = 'maxpool_2')(cnv6)
        fltn = Flatten(name = 'flatten_layer')(mxp3)
        FC1 = Dense(50, name = 'FC_1')(fltn)
        FC1 = LeakyReLU(alpha = 0.3, name = 'leaky_ReLu_1')(FC1)
        FC2 = Dense(50, name = 'FC_2')(FC1)
        FC2 = LeakyReLU(alpha = 0.3, name = 'leaky_ReLu_2')(FC2)
        output = Dense(num_classes, activation = 'softmax', name = 'output_layer')(FC2)
        model = Model(inputs = inp, outputs = output, name = 'WheatClassifier_CNN_'+str(model_id))
        model.summary()

    elif model_id == 3:
        image_size = target_size[0]  # We'll resize input images to this size
        patch_size = 10  # Size of the patches to be extract from the input images
        num_patches = (image_size // patch_size) ** 2
        projection_dim = 64
        num_heads = 4
        transformer_units = [
            projection_dim * 2,
            projection_dim]  # Size of the transformer layers
        transformer_layers = 2
        mlp_head_units = [50, 50]
        model = create_vit_classifier((*target_size, 3),
                                      patch_size,
                                      num_patches,
                                      projection_dim,
                                      transformer_units,
                                      transformer_layers, 
                                      'WheatClassifier_VIT_'+str(model_id),
                                      num_heads, mlp_head_units, num_classes)
        model.summary()
        
    elif model_id == 4:
        image_size = target_size[0]  # We'll resize input images to this size
        patch_size = 10  # Size of the patches to be extract from the input images
        num_patches = (image_size // patch_size) ** 2
        projection_dim = 64
        num_heads = 4
        transformer_units = [
            projection_dim * 2,
            projection_dim]  # Size of the transformer layers
        transformer_layers = 4
        mlp_head_units = [50, 50]
        model = create_vit_classifier((*target_size, 3),
                                      patch_size,
                                      num_patches,
                                      projection_dim,
                                      transformer_units,
                                      transformer_layers,
                                      'WheatClassifier_VIT_'+str(model_id),
                                      num_heads, mlp_head_units, num_classes)

        model.summary()


    return model
