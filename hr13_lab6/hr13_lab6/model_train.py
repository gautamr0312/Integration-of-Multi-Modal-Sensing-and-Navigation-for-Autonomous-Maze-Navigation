#!/usr/bin/env python3

############################
# Program to train the model
# Authors:
# Himanshu V
# Gautam Ramesh
############################

import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten

imageDirectory = './Combine_imgs/'

SIZE = (30,30)

with open(imageDirectory + 'train.txt', 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

train_data = []

for i in range(len(lines)):
    if lines[i][0][0] == "f":
        train_data.append(np.array([np.array(cv2.resize(cv2.imread(imageDirectory +lines[i][0]+".png",1),SIZE))]))
    else:
        train_data.append(np.array([np.array(cv2.resize(cv2.imread(imageDirectory +lines[i][0]+".jpg",1),SIZE))]))
        
train_data = np.array(train_data)
train_data = np.reshape(train_data, (train_data.shape[0], 30, 30, 3))
train_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])

# Train Test Spilt
X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# CNN Model Building
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(6, activation='softmax'))


#Compilation of the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model Training
eps = 50
anc = model.fit(X_train, y_train, batch_size=64, epochs=eps, validation_data=(X_test, y_test))
model.save(imageDirectory + "model.h5")

# Accuracy and Loss Plots
plt.figure(0)
plt.plot(anc.history['accuracy'], label='training accuracy')
plt.plot(anc.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(anc.history['loss'], label='training loss')
plt.plot(anc.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()



