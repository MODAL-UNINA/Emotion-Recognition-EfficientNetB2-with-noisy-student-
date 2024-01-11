#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Lun Ott 16 15:42:19 2023

@author: Diletta
"""

# %%
import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
# import cv2
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB7, EfficientNetV2M
from tensorflow.keras.layers import Dropout, Dense, Input, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical, plot_model


CVD = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = CVD

# check visible devices
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# set seed for reproducibility
np.random.seed(42)
# tf set seed
tf.random.set_seed(42)
# %%
model_to_train = 'B2' # 'B0' or 'B7'
# %%

# Auxiliary function for data distribution visualization
def data_visualization(classes, data):
    """
    Vizualize the distribution of the dataset.
  
    This function presents the number of samples each category has
    through a bar plot.

    Args:
        classes (list): List of the emotion-categories
        data (list): List of the number of images per category 
 
    Returns:
        No value
    """

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(classes, data)
    ax.set(title="Dataset's distribution for each Emotion class")
    ax.set(xlabel="Emotions", ylabel="#Images")
    ax.grid()


def load_data(data_path, data_name, model):
    """
    Load the dataset to the memory.
  
    This function takes a data path and loads all images (along with their
    corresponding labels) as numpy arrays per category to the memory.
    Subsequently, the labels are one-hot encoded. Before the images, labels
    are returned, the distribution of the dataset is presented.

    Args:
        data_path (str): The path of the dataset's whereabouts
   
    Returns:
        data (numpy.ndarray): The images of the dataset
        labels (numpy.ndarray): The labels of each image
    """

    subfolders_ck = os.listdir(data_path)

    print("[INFO] Dataset Loading...\n")

    img_data_list=[]
    labels_list = []
    num_images_per_class = []

    for category in subfolders_ck: # break
        img_list=os.listdir(data_path +'/'+ category)
        
        print('Loading :', len(img_list), 'images of category', category)
        for img in img_list: # break
            # Load an image from this path
            # pixels=cv2.imread(data_path + '/'+ category + '/'+ img ) 
            # read with PIL with three channels
            pixels = np.array(Image.open(data_path + '/'+ category + '/'+ img ).convert('RGB'))
            # plt.imshow(pixels)
            # face_array=cv2.resize(pixels, None, fx=1, fy=1,interpolation = cv2.INTER_CUBIC)
            # resize with PIL 
            if model == 'B0':
                face_array = np.array(Image.fromarray(pixels).resize((224, 224), resample=Image.BICUBIC, box = None))
            elif model == 'B1':
                face_array = np.array(Image.fromarray(pixels).resize((240, 240), resample=Image.BICUBIC, box = None))
            elif model == 'B2':
                face_array = np.array(Image.fromarray(pixels).resize((260, 260), resample=Image.BICUBIC, box = None))
            elif model == 'B7':
                face_array = np.array(Image.fromarray(pixels).resize((600, 600), resample=Image.BICUBIC, box = None))
            elif model == 'V2M':
                face_array = np.array(Image.fromarray(pixels).resize((128, 128), resample=Image.BICUBIC, box = None))
            # plt.imshow(face_array)
        
            img_data_list.append(face_array)          
            labels_list.append(category)

        num_images_per_class.append(len(img_list))


    if data_name == 'FER2013':
        emo_dict = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
        labels = [emo_dict[i] for i in labels_list]
    elif data_name == 'CK+48':
        emo_dict = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sadness': 4, 'surprise': 5, 'contempt': 6}
        labels = [emo_dict[i] for i in labels_list]

    
    labels = to_categorical(labels, 7)

    # association of labels with emotions

    # data_visualization(subfolders_ck, num_images_per_class)

    data = np.array(img_data_list)
    # Dataset Summary
    print("\nTotal number of uploaded data: ", data.shape[0],
          ", with data shape", (data.shape[1],data.shape[2],data.shape[3]))

    return data, labels, emo_dict
     

# %% 

path = os.getcwd()

#data_path_jaffe = "/content/drive/MyDrive/Data/jaffe"
data_path_ck = path + "/CK+48"
data_CK48, labels_CK48, emodict_CK48 = load_data(data_path_ck, 'CK+48', model_to_train)

data_path_ck = path + "/dati/FER2013/train"
data_FER_train, labels_FER_train, emodict_FERtrain = load_data(data_path_ck, 'FER2013', model_to_train)

data_path_ck =  path + "/dati/FER2013/test"
data_FER_test, labels_FER_test, emodict_FERtest = load_data(data_path_ck, 'FER2013', model_to_train)

data_path_ck =  path + "/Dati_disgust"
data_disgust, labels_disgust, emodict_disgust = load_data(data_path_ck, 'FER2013', model_to_train)

# np.unique(np.argmax(labels_disgust, axis=1), return_counts=True)
#%%

# from CK48 images and labels, remove the images and labels of contempt

data_CK48 = np.delete(data_CK48, np.where(labels_CK48[:,6] == 1), axis=0)
labels_CK48 = np.delete(labels_CK48, np.where(labels_CK48[:,6] == 1), axis=0)

print(data_CK48.shape)
print(labels_CK48.shape)

#check new distribution
np.unique(np.argmax(labels_CK48, axis=1), return_counts=True)
# print for each emotion the number of images using also the dictionary
for key, value in emodict_CK48.items():
    if value in np.unique(np.argmax(labels_CK48, axis=1)):
        print(key, ' : ', np.unique(np.argmax(labels_CK48, axis=1), return_counts=True)[1][value])

# combine the three datasets in one

dataset_merge = np.concatenate((data_CK48, data_FER_train, data_FER_test, data_disgust), axis=0)
labels_merge = np.concatenate((labels_CK48, labels_FER_train, labels_FER_test, labels_disgust), axis=0)

print(dataset_merge.shape)
print(labels_merge.shape)

# randomly shuffle data in the same way for both data and labels
np.random.seed(42)
data = zip(dataset_merge, labels_merge)
data = list(data)
np.random.shuffle(data)
dataset_merge, labels_merge = zip(*data)

# plot 100 random images from the dataset with their labels
fig, axes = plt.subplots(10,10, figsize=(15,15))
for i,ax in enumerate(axes.flat):
    ax.imshow(dataset_merge[i], interpolation='nearest')
    ax.set_axis_off()
    ax.set_title(f'{np.argmax(labels_merge[i])} = {list(emodict_FERtest.keys())[list(emodict_FERtest.values()).index(np.argmax(labels_merge[i]))]}')

# emodict_CK48
# emodict_FERtest
# emodict_FERtrain

dataset_merge = np.array(dataset_merge)
labels_merge = np.array(labels_merge)

# plot 10 images where label is 1
# fig, axes = plt.subplots(2,5, figsize=(15,15))
# for i,ax in enumerate(axes.flat):
#     ax.imshow(dataset_merge[np.where(np.argmax(labels_merge, axis=1) == 1)][i], interpolation='nearest')
#     ax.set_axis_off()
#     ax.set_title(f'{np.argmax(labels_merge[np.where(np.argmax(labels_merge, axis=1) == 1)][i])} = {list(emodict_FERtest.keys())[list(emodict_FERtest.values()).index(np.argmax(labels_merge[np.where(np.argmax(labels_merge, axis=1) == 1)][i]))]}')

np.unique(np.argmax(labels_merge, axis=1), return_counts=True)
# %%

# split the dataset in train, validation and test

X_train, X_test, y_train, y_test = train_test_split(dataset_merge, labels_merge, test_size=0.30, random_state=42, stratify=labels_merge)

X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.50, random_state=42, stratify=y_test)

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

# %%
# # plot 100 random images from the dataset with their labels
# fig, axes = plt.subplots(10,10, figsize=(15,15))
# for i,ax in enumerate(axes.flat):
#     ax.imshow(X_train[i], interpolation='nearest')
#     ax.set_axis_off()
#     ax.set_title(f'{np.argmax(y_train[i])} = {list(emodict_FERtest.keys())[list(emodict_FERtest.values()).index(np.argmax(y_train[i]))]}')

# %%

# Map the emotion-categories
# mapping = {0:'anger', 1:'disgust', 2:'fear', 3:'happy', 4:'sadness', 5:'surprise'}


# Initialize the training data augmentation object
trainAug = ImageDataGenerator(rotation_range=15,
                              zoom_range=0.15,
                              width_shift_range=0.2,
                              brightness_range=(.6, 1.2),
                              shear_range=.15,
                              height_shift_range=0.2,
                              horizontal_flip=True,
                              fill_mode="nearest")

# # Randomly vizualize some augmented samples
# mapping = emodict_FERtrain
# plt.figure(figsize=(10,10))
# for i, (image, label) in enumerate(trainAug.flow(X_train, y_train, batch_size=1)):
#     if i == 9:
#         break

#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(np.squeeze(image)/255.)
#     plt.title(f'{np.argmax(label)} = {list(mapping.keys())[list(mapping.values()).index(np.argmax(label))]}')
#     plt.axis("off")

#%%

# convert weights:  python efficientnet_weight_update_util.py --model b0 --notop --ckpt /home/diletta/Projects/ER_train/noisy_student_efficientnet-b0/model.ckpt --o efficientnetb0_notop.h5


from keras import regularizers
#tf.keras.applications.efficientnet.preprocess_input
# /home/diletta/Projects/ER_train/noisy_student_efficientnet-b0/efficientnetb0_notop.h5

if model_to_train == 'B0':
    input_size = (224, 224, 3)
    student_weights = '/home/diletta/Projects/ER_train_bk/noisy_student_efficientnet-b0/efficientnetb0_notop.h5'
    mod = EfficientNetB0
elif model_to_train == 'B1':
    input_size = (240, 240, 3)
    student_weights = '/home/diletta/Projects/ER_train_bk/noisy_student_efficientnet-b1/efficientnetb1_notop.h5'
    mod = EfficientNetB1
elif model_to_train == 'B2':
    input_size = (260, 260, 3)
    student_weights = '/home/diletta/Projects/ER_train_bk/noisy_student_efficientnet-b2/efficientnetb2_notop.h5'
    mod = EfficientNetB2
elif model_to_train == 'B7':
    input_size = (600, 600, 3)
    student_weights = '/home/diletta/Projects/ER_train_bk/noisy_student_efficientnet-b7/efficientnetb7_notop.h5'
    mod = EfficientNetB7
elif model_to_train == 'V2M':
    input_size = (128, 128, 3)
    student_weights = 'imagenet'
    mod = EfficientNetV2M
# %%


# create model

def build_model():
    """
    Create the new model.
  
    This function loads a pre-trained EfficientNetB0 model on Imagenet,
    adds an Input layer at the beggining, and 3 more layers at the end
    (i.e GlobalAveragePooling2D, Dropout, Dense).

    Args:
        No value
   
    Returns:
        model (tensorflow.python.keras.engine.functional.Functional): The compiled model
    """
    
    inputs = Input(shape=input_size)
    # preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    # inputs = preprocess_input(inputs)
    base_model = mod(include_top=False, weights=student_weights,
                                drop_connect_rate=0.33, 
                                input_tensor=inputs,
                                input_shape=input_size)
    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dropout(.2, name="top_dropout")(x) #0.5
    # add regularized dense layer
    # x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01),
    #             activity_regularizer=regularizers.l1(0.01))(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                activity_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(x)
    # x = Dropout(.3, name="top_dropout2")(x) #0.5
    # x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01),
    #             activity_regularizer=regularizers.l1(0.01))(x)
    # x = Dropout(.3, name="top_dropout3")(x) #0.5
    outputs = Dense(7, activation='softmax')(x)
    model = Model(inputs, outputs)

    model.compile(optimizer=Adam(learning_rate=1e-2),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Create an object of the model
model = build_model()

# %%

from contextlib import redirect_stdout

with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

#print(model.summary())
# plot_model(model, to_file='model.png')

# %%

# model train

EPOCHS = 200
batch_size = 100
print(batch_size)
# check directory
# dir = os.path.dirname(os.path.realpath(__file__))
filepath = f"weights_ER_{model_to_train}.hdf5"


checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlystopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', restore_best_weights=True)
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.25, min_lr=1e-7, verbose=1)


callbacks = [checkpoint, earlystopping, rlrop]

# put together X_train and X_test
X_train = np.concatenate((X_train, X_test), axis=0)
y_train = np.concatenate((y_train, y_test), axis=0)


# compute class weights to balance the dataset
weight_for_class = {}
for i in range(7):
    weight_for_class[i] = (1 / np.sum(np.argmax(y_train, axis=1) == i)) * (X_train.shape[0]) / 7.0
# print(weight_for_class)


print(f"[INFO] training network for {EPOCHS} epochs...\n")
hist = model.fit(trainAug.flow(X_train, y_train, batch_size=batch_size, seed=42),
                 steps_per_epoch=len(X_train) // batch_size,
                 validation_data=(X_val, y_val),
                 epochs=EPOCHS, callbacks=callbacks,
                    class_weight=weight_for_class)

# hist = model.fit(X_train, y_train,
#                  steps_per_epoch=len(X_train) // batch_size,
#                  validation_data=(X_val, y_val),
#                  epochs=EPOCHS, callbacks=callbacks)

# %%

# # Clear the values of previous plot
fig = plt.figure(figsize=(15, 7))

# Plot training & validation accuracy values
fig.add_subplot(121)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(f'Training & Validation Accuracy Plot {model_to_train} ')
# plt.show()
# Plot training & validation loss values
fig.add_subplot(122)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(f'Training & Validation Loss Plot {model_to_train} ')
# plt.show()
#%% 



