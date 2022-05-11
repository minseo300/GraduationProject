
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


# 학습 데이터 불러오기
def loadData(trainData_directory,classes):
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = np.load("/content/drive/MyDrive/bottom_length_2classes.npy", allow_pickle=True)
    global classes_name
    classes_name=[]
    for i in range(classes):
        classes_name.append(str(i))

# CNN 모델
def cnnModel(imgSize,classes):
    global model
    model= Sequential()
    model.add(Convolution2D(imgSize, (3, 3), padding='same',
                            input_shape=(imgSize, imgSize, 3)))
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

# CNN 모델 build, 학습
def buildCompileModel(checkPoint_directory,imgSize):
    EPOCHS = 100
    INIT_LR = 1e-4
    BS = 64  # 배치 사이즈
    N_TRAIN = X_train.shape[0]
    N_TEST = X_test.shape[0]
    steps_per_epoch = N_TRAIN / BS
    validation_steps = N_TEST / BS

    model_checkpoint = ModelCheckpoint(
        filepath=checkPoint_directory+'_{epoch}-{val_loss:.2f}-{val_accuracy:.2f}.h5',
        monitor='val_loss', save_best_only=True, verbose=1)

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    model.build((None, imgSize, imgSize, 3))
    model.summary()

    H = model.fit(X_train, y_train, epochs=EPOCHS, verbose=1, validation_data=(X_test, y_test),
                  steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, callbacks=model_checkpoint)
    # 성능 확인
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

#데이터 불러오기
data_directory="/content/drive/MyDrive/bottom_length_2classes.npy"
loadData(data_directory)

#CNN 모델(이미지 사이즈, 클래스 개수)
cnnModel(32,2)

#학습(모델 저장 경로)
buildCompileModel('/content/drive/MyDrive/bottom_length_2classes')