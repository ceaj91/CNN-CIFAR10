# -*- coding: utf-8 -*-
"""saving_weights_script.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12IchgT09tri5mn-OLikUxyeKJzwmZg0d
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.layers import Dropout
from tensorflow.keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from keras.models import Model
from keras.models import load_model
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train=x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define CNN architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
#model.add(Dropout(0.2)) #izbaci
model.add(layers.Dense(512, activation='relu'))
#model.add(Dropout(0.2)) #izbaci
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.load_weights('model.h5')
for layer in model.layers:
    if len(layer.get_weights()) > 0:
      print(layer.name)
      for w in layer.get_weights():
            print(w.shape, w.dtype)

layer1 = model.get_layer('conv2d').get_weights()
filter_weights = layer1[0]
filter_bias = layer1[1]
target = open("conv1_filters.txt",'w')
for filter in range(32):
    for channel in range(3):
      np.savetxt(target,filter_weights[:,:,channel,filter], fmt='%.18f',delimiter=' ')
target.close()
target = open("conv1_bias.txt",'w')
np.savetxt(target,filter_bias, fmt='%.18f',delimiter=' ')

layer2 = model.get_layer('conv2d_1').get_weights()
filter_weights = layer2[0]
filter_bias = layer2[1]
target = open("conv2_filters.txt",'w')
for filter in range(32):
    for channel in range(32):
      np.savetxt(target,filter_weights[:,:,channel,filter], fmt='%.18f',delimiter=' ')
target.close()
target = open("conv2_bias.txt",'w')
np.savetxt(target,filter_bias, fmt='%.18f',delimiter=' ')

layer3 = model.get_layer('conv2d_2').get_weights()
filter_weights = layer3[0]
filter_bias = layer3[1]
target = open("conv3_filters.txt",'w')
for filter in range(64):
    for channel in range(32):
      np.savetxt(target,filter_weights[:,:,channel,filter], fmt='%.18f',delimiter=' ')
target.close()
target = open("conv3_bias.txt",'w')
np.savetxt(target,filter_bias, fmt='%.18f',delimiter=' ')
target.close()

layer4 = model.get_layer('dense').get_weights()
filter_weights = layer4[0]
filter_bias = layer4[1]
target = open("dense1_weights.txt",'w')
np.savetxt(target,filter_weights, fmt='%.18f',delimiter=' ')
target.close()

target = open("dense1_bias.txt",'w')
np.savetxt(target,filter_bias, fmt='%.18f',delimiter=' ')
target.close()

layer5 = model.get_layer('dense_1').get_weights()
filter_weights = layer5[0]
filter_bias = layer5[1]
target = open("dense2_weights.txt",'w')
np.savetxt(target,filter_weights, fmt='%.18f',delimiter=' ')
target.close()

target = open("dense2_bias.txt",'w')
np.savetxt(target,filter_bias, fmt='%.18f',delimiter=' ')
target.close()