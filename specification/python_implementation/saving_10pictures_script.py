# -*- coding: utf-8 -*-
"""saving_10pictures_script.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16v-lRoZ9rbsV78bhbERae7WvLK-O24DW
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
import numpy as np

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train=x_train.astype('float32')
x_test = x_test.astype('float32')
slika=x_test[0,:,:,:].astype('int')
# Normalize pixel values between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train)
y_test = y_test.astype('int')

target = open("slike.txt",'w')
for picture in range(10):
  for channel in range(3):
        np.savetxt(target,x_test[picture,:,:,channel], fmt='%.18f',delimiter=' ')
target.close()

target = open("labele.txt",'w')
for label in range(10):
  np.savetxt(target,y_test[label],fmt='%d')
target.close()