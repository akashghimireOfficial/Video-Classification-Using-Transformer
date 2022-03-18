from tensorflow_docs.vis import embed
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Dropout,GlobalMaxPool1D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical,plot_model

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

import create_dataset


MAX_SEQ_LENGTH = 20
NUM_FEATURES = 1024
IMG_SIZE = 128

EPOCHS = 5


#features,labels=create_dataset.create_dataset()
#features=np.array(features)
#np.save('features.npy',features)
#labels=np.array(labels)
#np.save('labels.npy',labels)

seed_constant = 50
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

labels=np.load('labels.npy')
features=np.load('features.npy')
num_class=len(np.unique(labels))

labels=to_categorical(labels,num_classes=num_class)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

class PositionalEmbedding(layers.Layer):
    def __init__(self,sequence_length,output_dim,**kwargs):
        super().__init__(**kwargs)
        self.position_embedding=layers.Embedding(input_dim=sequence_length,
                                                output_dim=output_dim)
        
        self.sequence_length=sequence_length
        self.output_dim=output_dim
        
    def call(self,inputs):
        length=tf.shape(inputs)[1]
        positions=tf.range(start=0,limit=length,delta=1)
        embedded_positions=self.position_embedding(positions)
        return inputs +embedded_positions
    
    def compute_mask(self,inputs,mask=None):
        mask=tf.reduce_any(tf.cast(inputs,'bool'),axis=-1)
        return mask

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = tf.keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)     
        
        
        
def get_compiled_model():
    sequence_length=MAX_SEQ_LENGTH
    embed_dim=NUM_FEATURES
    dense_dim=4
    num_heads=1
    classes=num_class
    lr=0.001
    inputs=tf.keras.Input(shape=(sequence_length,embed_dim))
    x=PositionalEmbedding(sequence_length,embed_dim,name='Frame_positional_embedding')(inputs)
    x=TransformerEncoder(embed_dim,dense_dim,num_heads,name='TransformerEncoder')(x)
    x=GlobalMaxPool1D()(x)
    x=Dropout(0.5)(x)
    outputs=Dense(num_class,activation='softmax')(x)
    model=tf.keras.Model(inputs,outputs,name='vit')
    model.compile(optimizer=Adam(learning_rate=lr),loss='categorical_crossentropy',metrics=['accuracy'])
    return model
               
