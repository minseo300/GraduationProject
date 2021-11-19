import PIL
from PIL import Image
import os,re,glob
import cv2
import numpy as np
import json
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
import os
import glob
import shutil as sh
from sklearn.model_selection import train_test_split
from PIL import Image
import os,re,glob
import cv2
import numpy as np
import json

#  이미지 증식 (이미지 부족한 라벨 데이터 증식)
datagen=ImageDataGenerator(rotation_range=20,
                           width_shift_range=0.1,
                           height_shift_range=0.1,
                           vertical_flip=True,
                           horizontal_flip=True,
                           fill_mode='nearest')


filename_in_dir = []

for root, dirs, files in os.walk('D:/crawling_data/5'):
    for fname in files:
        full_fname = os.path.join(root, fname)
        filename_in_dir.append(full_fname)

for file_image in filename_in_dir:
    print(file_image)
    img = load_img(file_image)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0

    for batch in datagen.flow(x, save_to_dir='D:/crawling_data/5', save_prefix='_gen_', save_format='JPG'):
        i += 1
        if i > 5:
            break

###################################################################################################

#dataset npy 파일 생성 후 저장
# X,y=np.load("D:/numpy data/top_categories_test.npy",allow_pickle=True)
#
# print(X.shape)

data_dir='D:/crawling_data'
classes=5
categories=['1','2','3'] # crawling_data <긴팔-0 / 맨투맨-1 / 민소매-2 / 반팔-3 / 셔츠-4 / 후드-5> /   sleeve_length_data 0: 민소매, 1:반팔, 2: 긴팔

image_size=32

pixels=image_size*image_size*3

X=[]
y=[]
count=0
for idx,cat in enumerate(categories):
    label=[0 for i in range(classes)]
    label[idx]=1
    count=0

    image_dir=data_dir+'/'+cat
    print('image dir: ',image_dir)
    files=glob.glob(image_dir+'/*.JPG')

    for i,f in enumerate(files):
        #print("f: ",f)
        if count>=10000:
            break
        try:
            img = Image.open(f).convert('RGB')
            img=img.resize((32,32))
            img=np.array(img)
            X.append(img/255.)
            y.append(label)
            count+=1
        except PIL.UnidentifiedImageError:
            print('error f: ',f)

X=np.array(X)
y=np.array(y)

print("X shape: ",X.shape)
print("y shape: ",y.shape)
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.3,shuffle=True,random_state=42)

xy=(X_train,X_test,y_train,y_test)
# print('X_train shape: ',X_train.shape)
# print('X_test shape: ',X_test.shape)
# print('y_train shape: ',y_train.shape)
# print('y_test shape: ',y_test.shape)

print('X shape: ',X.shape)
np.save("D:/numpy data/sleeve_length.npy",xy)