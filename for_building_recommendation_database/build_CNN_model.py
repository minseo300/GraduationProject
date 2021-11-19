
# from keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Flatten, Convolution2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from PIL import Image
import os,re,glob
import cv2
import numpy as np
import json
from tensorflow.keras import optimizers
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


#데이터 불러오기
X_train,X_test,y_train,y_test=np.load("/content/drive/MyDrive/bottom_length_2classes.npy",allow_pickle = True)
class_name=["0","1"]
#class_name=["0","1","2","3","4"]
EPOCHS = 100
INIT_LR = 1e-4
BS = 64 #배치 사이즈
N_TRAIN=X_train.shape[0]
N_TEST=X_test.shape[0]
steps_per_epoch=N_TRAIN/BS
validation_steps=N_TEST/BS
steps_per_epoch=N_TRAIN/BS
validation_steps=N_TEST/BS

classes=2
#classes=4

print("X_train shape: ",X_train.shape)
print("y_train shape: ",y_train.shape)
print("X_test shape: ",X_test.shape)
print("y_test shape: ",y_test.shape)

model = Sequential()
#Conv2D

model.add(Convolution2D(32, (3, 3), padding='same',
                 input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(classes))
model.add(Activation('softmax'))


# model_json=model.to_json()
# with open("/content/drive/MyDrive/top_categories_except_longsleeve.json","w") as json_file:
#   json_file.write(model_json)
# 콜백 함수 적용 부분
#early_stopping=EarlyStopping(monitor='val_loss',patience=100)
model_checkpoint=ModelCheckpoint(filepath='/content/drive/MyDrive/bottom_length_2classes_{epoch}-{val_loss:.2f}-{val_accuracy:.2f}.h5',monitor='val_loss',save_best_only=True,verbose=1)

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
model.build((None,32,32,3))
model.summary()




#성능 확인
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])