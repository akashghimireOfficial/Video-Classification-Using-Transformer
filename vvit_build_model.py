from tensorflow_docs.vis import embed
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Dropout,GlobalMaxPool1D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.optimizers import Adam

import pandas as pd
import numpy as np
import imageio
from numpy import random
import cv2
import pafy
import os

import datetime as dt 
from collections import deque
from moviepy.editor import *
import pafy
import random
import time



labels=np.load('vivit_labels.npy')
features=np.load('vivit_features.npy')

INPUT_SHAPE = (64,64,64, 3)
NUM_CLASSES=len(np.unique(labels))



LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

EPOCHS = 60

PATCH_SIZE = (8, 8, 8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2


# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8




#features,labels=create_dataset.create_dataset()
#features=np.array(features)
#np.save('features.npy',features)
#labels=np.array(labels)
#np.save('labels.npy',labels)

seed_constant = 50
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

##Creating training and testing data

labels=to_categorical(labels,num_classes=NUM_CLASSES)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)


class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches

    
class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens    
    
    
    





def create_vivit_classifier(
    
    
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES,
):
    tubelet_embedder=TubeletEmbedding(embed_dim=embed_dim,patch_size=PATCH_SIZE)
    positional_encoder=PositionalEncoder(embed_dim=embed_dim)
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = tubelet_embedder(inputs)
    # Encode patches.
    encoded_patches = positional_encoder(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = tf.keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=embed_dim, activation=tf.nn.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)

    # Classify outputs.
    outputs = layers.Dense(units=num_classes, activation="softmax")(representation)

    # Create the Keras model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

               


































