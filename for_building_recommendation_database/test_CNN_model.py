# 저장한 모델 불러와서 테스트
from keras.layers import Flatten, Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization
from PIL import Image
import os,re,glob
import cv2
import numpy as np
import json
from keras import optimizers
from tensorflow.python import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
from keras.models import load_model
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import load_model
import matplotlib.pyplot as plt


# prepare the data to predict
test_num=Image.open('/content/drive/MyDrive/민소매1.jpg').convert('RGB')
test_num=np.array(test_num)
test_num = test_num.astype('float32') / 255.
plt.imshow(test_num)
test_num = test_num.reshape((1,32, 32,3))

# load the model
model=load_model('/content/drive/MyDrive/sleeve_length_121-0.13-0.96.h5') # sleeve_length / top_categories_except_longsleeve / bottom_length_276-0.37-0.88 / bottom_categories_285-0.31-0.90
model.summary()

# predict the class
print("The Answer is ",np.argmax(model.predict(test_num)))