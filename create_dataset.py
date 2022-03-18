import os 
from glob import glob 
import tensorflow as tf 
from tensorflow.keras.layers import Layer,Dropout,Dense,BatchNormalization,Flatten
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



import datetime as dt 
from collections import deque
from moviepy.editor import *
import pafy
import random
import time


seed_constant = 50
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

data_set_dire='UCF-101'
CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
seq_len=20
resized_height,resized_width=128,128
num_features=1024


def extract_frames(video_dire):
    extracted_frames=[]
    video_reader=cv.VideoCapture(video_dire)
    total_frames=int(video_reader.get(cv.CAP_PROP_FRAME_COUNT))
    skip_frames=max(int(total_frames/seq_len),1)
    for frame_counter in range(total_frames):
        video_reader.set(cv.CAP_PROP_POS_FRAMES,frame_counter*skip_frames)
        sucess,frame=video_reader.read()
        if not sucess:
            break
        resized_frame=cv.resize(frame,(resized_width,resized_height))
        extracted_frames.append(resized_frame)
    extracted_frames=np.array(extracted_frames)
    return extracted_frames    


def features_extraction_model():
    feature_extractor=DenseNet121(include_top=False,weights='imagenet',
                        pooling='avg',
                        input_shape=(resized_height,resized_width,3))
    inputs=tf.keras.Input((resized_height,resized_width,3))
    preprocessed=preprocess_input(inputs)
    outputs=feature_extractor(preprocessed)
    model=tf.keras.Model(inputs,outputs,name='feature_extractor')
    return model
    
    
    
    
feature_extractor_model=features_extraction_model()

def create_dataset():
    labels=[]
    features=[]
    for idx,className in enumerate(CLASSES_LIST):
        video_dires_list=glob(os.path.join(data_set_dire,className)+"/*")
        for video_dire in video_dires_list:
            extracted_frames=extract_frames(video_dire)
            if len(extracted_frames)==20:
                print(1)
                get_features=feature_extractor_model.predict(extracted_frames)
                features.append(get_features)
                labels.append(idx)
                
    return features,labels
    



