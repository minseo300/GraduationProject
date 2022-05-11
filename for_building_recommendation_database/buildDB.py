import boto3
from botocore.client import Config
import PIL
from PIL import Image
import os,re,glob
import cv2
import numpy as np
import json
#from tensorflow.python import keras
#from keras.models import Sequential
# from keras.layers import Dropout, Activation, Dense
# from keras.layers import Flatten, Convolution2D, MaxPooling2D
#import tensorflow as tf
import matplotlib.pyplot as plt
#from keras.models import load_model
from sklearn.model_selection import train_test_split
#from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
import os
import glob
import shutil as sh
from sklearn.model_selection import train_test_split
from PIL import Image
import os,re,glob
import cv2
import numpy as np
import json
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pymysql
import color
from sklearn.cluster import KMeans

# dictionary about classifying labels
outer_cate_dict={0:'블레이저', 1:'트레이닝', 2:'무스탕',3:'트렌치코트',4:'코트',5:'트러커 자켓', 6:'라이더', 7: '블루종',8:'롱패딩', 9:'숏패딩',10:'가디건',11:'후드집업'}
top_cate_dict={0:'맨투맨', 1:'민소매', 2:'반팔', 3:'셔츠', 4:'후드티'}
sleeve_length_dict={0:'긴팔', 1:'민소매', 2:'반팔'}
bottom_cate_dict={ 0:'스커트', 1:'레깅스', 2:'숏팬츠', 3:'슬랙스', 4:'조거팬츠'}
bottom_length_dict={0:'미니',1:'맥시'}
bottom_fit_dict={0:'와이드팬츠', 1:'스키니진', 2:'일자바지', 3:'부츠컷'}
print_dict={0: '무지',1:'스트라이프',2:'그래픽'}

# mysql table columns names
INFO_COLUMNS_NAME='ID, outer_cate, outer_color, outer_link, top_cate, top_sleevelength, top_color, top_print,top_link, bottom_cate, bottom_color, bottom_length, bottom_link, onepiece_sleevelength, onepiece_length, onepiece_color, onepiece_link, temperature_section, gender'
IMAGE_COLUMNS_NAME='ID, image, category'

# connect with mysql
# db=pymysql.connect(host="hostname ip주소", user="minseo",password="minseopw", db="db 이름")
# 탄력적 IP 주소: TODO
db=pymysql.connect(host="52.79.59.24", user="minseo",password="minseopw", db="SmartMirror",charset='utf8')
curs=db.cursor()

def classify_color(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))  # height, width 통합

        k = 3
        clt = KMeans(n_clusters=k)
        clt.fit(image)
        hist = color.centroid_histogram(clt)
        back = 0
        for i in range(3):
            print(i, hist[i], color.find_color(clt.cluster_centers_[i]))
            if hist[i] < 0.1:
                continue
            if color.find_color(clt.cluster_centers_[i]) == "흰색":
                if back == 1:
                    print("흰색")
                    break
                back = 1
            else:
                answer_color = color.find_color(clt.cluster_centers_[i])
                print(answer_color)
                break
    except:
        answer_color='null'

    return answer_color

# s3 bucket - Todo: KEEP IT SECURE
BUCKET_NAME= 'smartmirror-bucket'
ACCESS_KEY_ID='AKIA462PQS3AXQ5D2BNE'
ACCESS_SECRET_KEY='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'

# root directory
root_dir="C:/Users/lmslm/Desktop/dd/"
folders=os.listdir(root_dir)
s3=boto3.client('s3')
response=s3.list_buckets()
print(response)

# load cnn models
bottom_categories=load_model('bottom_categories.h5')
bottom_length=load_model('bottom_length.h5')
outer_categories=load_model('outer_categories.h5')
top_categories=load_model('top_categories.h5')
sleeve_length=load_model('sleeve_length.h5')
bottom_fit=load_model('bottom_fit.h5')
print_=load_model('print.h5')

for style_name in folders: # style_name - casual, formal, ...
    # folder directory
    print("Style: ",style_name)
    folder_dir=root_dir+style_name
    styles=os.listdir(folder_dir)
    for style in styles: # style - 1,2,3, ...
        ID = int(style) # styling ID
        # initializing attributes
        outer_cate = 'null'
        outer_color = 'null'
        outer_link = 'null'
        top_cate = 'null'
        top_sleeve_length = 'null'
        top_color = 'null'
        print_cate = 'null'
        top_link = 'null'
        bottom_cate = 'null'
        bottom_color = 'null'
        bottom_length_cate = 'null'
        bottom_link = 'null'
        onepiece_length = 'null'
        onepiece_sleeve_length = 'null'
        onepiece_color = 'null'
        onepiece_link = 'null'
        gender='null'
        outer_temperature=0
        top_temperature=0
        bottom_temperature=0
        onepiece_temperature=0

        # styling directory
        styling_dir=root_dir+style_name+'/'+style
        print("styling_dir: ",styling_dir)
        styling=os.listdir(styling_dir)
        for file in styling: # styling_1_top.jpg, styling_1_bottom.jpg, ...
            print('file: ',file)
            file_directory=root_dir+style_name+'/'+style+'/'+file # image file directory
            if os.path.splitext(file)[1] == '.txt':
                replaced_file_name = file.replace('.', '_')
                category = replaced_file_name.split('_')
                text_file=open(file_directory,'r')
                # print(category)
                # print("category length: ", len(category))
                print("TEXT FILE: ",file)
                if category[2]=='outer':
                    outer_link=text_file.readline()
                    print('outer_link: ',outer_link)
                elif category[2]=='top':
                    top_link = text_file.readline()
                elif category[2]=='bottom':
                    bottom_link = text_file.readline()
                elif category[2]=='onepiece':
                    onepiece_link = text_file.readline()
                text_file.close()
            else:
                # Image
                image = Image.open(file_directory).convert('RGB')
                image.show()

                # Todo: s3에 이미지 업로드
                s3=boto3.resource('s3',aws_access_key_id=ACCESS_KEY_ID,aws_secret_access_key=ACCESS_SECRET_KEY,config=Config(signature_version='s3v4'))
                # s3.Bucket(BUCKET_NAME).put_object(Key=folder_name+'/'+file_name,Body=data,ContentType='image/jpg')
                # file_name=style_name+'_'+style+'_'+file
                # s3.upload_file(file_name,BUCKET_NAME,file_name)

                # image url in s3
                image_url = 'https://smartmirror-bucket.s3.ap-northeast-2.amazonaws.com/' + style_name + '/' + file

                # data is image file
                data=open(file_directory,'rb')
                replaced_file_name=file.replace('.','_')
                category=replaced_file_name.split('_')

                image_table_name = 'R_' + style_name + '_image'

                if len(category)==3:
                    print("styling")
                    # Todo: s3에 업로드한 이미지 url 정보 가져오기
                    values=(ID,image_url,"styling")
                    image_sql='insert into '+image_table_name+'('+IMAGE_COLUMNS_NAME+') values (%s,%s,%s)'
                    curs.execute(image_sql, values)
                    db.commit()
                    # gender
                    print('GENDER>>>')
                    gender = input("Gender - F for female / M for male / U for unisex : ")
                elif len(category)==4:

                    # Todo: s3에 업로드한 이미지 url 정보 가져오기
                    values = (ID, image_url ,category[2])
                    image_sql = 'insert into ' + image_table_name + '(' + IMAGE_COLUMNS_NAME + ') values (%s, %s, %s)'
                    curs.execute(image_sql,values)
                    db.commit()

                    resized_image = image.resize((32, 32))
                    resized_image=np.array(resized_image)
                    resized_image=resized_image.astype('float32')/255.


                    if category[2]=='outer':
                        print("<This is OUTER>")

                        # classify outer categories
                        print(">>>>>OUTER categories: 0-블레이저, 1-트레이닝, 2-무스탕, 3-트렌치코트, 4-코트, 5-트러커 자켓, 6-라이더, 7- 블루종, 8-롱패딩, 9-숏패딩, 10-가디건, 11-후드집업")

                        # plt.imshow(resized_image)
                        resized_image = resized_image.reshape((1, 32, 32, 3))
                        predict = np.argmax(outer_categories.predict(resized_image))
                        print("PREDICTION: ", predict)
                        user_input=int(input(">>>>>>>>If prediction is right, enter -1. If prediction is wrong, enter right class."))
                        if user_input==-1:
                            answer=predict
                        else:
                            answer=user_input

                        # determine temperature section using outer category
                        if answer==1 or answer==11 or answer==3:
                            outer_temperature=4
                        elif answer==0 or answer==5 or answer==6 or answer==7 or answer==11 :
                            outer_temperature=5
                        elif answer==2 or answer==4 or answer==9:
                            outer_temperature=6
                        elif answer==8:
                            outer_temperature=7

                        outer_cate = outer_cate_dict.get(answer)

                        # Todo: classify color
                        print(">>>>>COLOR")
                        outer_color= classify_color(file_directory)
                        print(outer_color)

                    elif category[2]=='top':
                        print("<This is TOP>")

                        # classify top categories
                        image_top_categories=image.resize((64,64))
                        image_top_categories = np.array(image_top_categories)
                        image_top_categories = image_top_categories.astype('float32') / 255.
                        # plt.imshow(image_top_categories)
                        print(
                            ">>>>>TOP categories: 0-맨투맨, 1-민소매, 2-반팔, 3-셔츠, 4-후드티")
                        image_top_categories = image_top_categories.reshape((1, 64, 64, 3))
                        predict = np.argmax(top_categories.predict(image_top_categories))
                        print("PREDICTION: ", predict)
                        user_input = int(
                            input(">>>>>>>>If prediction is right, enter -1. If prediction is wrong, enter right class."))
                        if user_input == -1:
                            answer = predict
                        else:
                            answer = user_input

                        #
                        if answer == 1:
                            top_temperature = 1
                        elif answer==2:
                            top_temperature=2
                        elif answer==0 or answer==4 or answer==3:
                            top_temperature=3

                        top_cate=top_cate_dict.get(answer)

                        # classify top sleeve length
                        print(
                            ">>>>>SLEEVE LENGTH: 0-긴팔, 1-민소매, 2-반팔")
                        # plt.imshow(resized_image)
                        resized_image = resized_image.reshape((1, 32, 32, 3))
                        predict = np.argmax(sleeve_length.predict(resized_image))
                        print("PREDICTION: ", predict)
                        user_input = int(
                            input(">>>>>>>>If prediction is right, enter -1. If prediction is wrong, enter right class."))
                        if user_input == -1:
                            answer = predict
                        else:
                            answer = user_input


                        top_sleeve_length=sleeve_length_dict.get(answer)

                        # classify print
                        image_top_categories = image.resize((64, 64))
                        image_top_categories = np.array(image_top_categories)
                        image_top_categories = image_top_categories.astype('float32') / 255.
                        print(
                            ">>>>>PRINT: 0-무지, 1-스트라이프, 2-그래픽")
                        # plt.imshow(resized_image)
                        image_top_categories = image_top_categories.reshape((1, 64, 64, 3))
                        predict = np.argmax(print_.predict(image_top_categories))
                        print("PREDICTION: ", predict)
                        user_input = int(
                            input(">>>>>>>>If prediction is right, enter -1. If prediction is wrong, enter right class."))
                        if user_input == -1:
                            answer = predict
                        else:
                            answer = user_input
                        print_cate=print_dict.get(answer)

                        # Todo: classify color
                        print(">>>>>COLOR")
                        top_color= classify_color(file_directory)
                        print(top_color)

                    elif category[2]=='bottom':
                        print("<This is BOTTOM>")

                        # classify bottom length
                        print(
                            ">>>>>BOTTOM length: 0-미니, 1-맥시")
                        # plt.imshow(resized_image)
                        resized_image = resized_image.reshape((1, 32, 32, 3))
                        predict = np.argmax(bottom_length.predict(resized_image))
                        print("PREDICTION: ", predict)
                        user_input = int(input(
                            ">>>>>>>>If prediction is right, enter -1. If prediction is wrong, enter right class."))
                        if user_input == -1:
                            answer = predict
                        else:
                            answer = user_input

                        if answer == 0:
                            bottom_temperature = 1
                        elif answer == 1:
                            bottom_temperature = 2
                        bottom_length_cate = bottom_length_dict.get(answer)

                        # classify bottom categories
                        print(
                            ">>>>>BOTTOM categories: 0-스커트, 1-레깅스, 2-숏팬츠, 3-슬랙스, 4-조거팬츠")
                        # plt.imshow(resized_image)
                        resized_image = resized_image.reshape((1, 32, 32, 3))
                        predict = np.argmax(bottom_categories.predict(resized_image))
                        print("PREDICTION: ",predict)
                        user_input = int(
                            input(">>>>>>>>If prediction is right, enter -1. If prediction is wrong, enter right class."))
                        if user_input == -1:
                            answer = predict
                        else:
                            answer = user_input
                        bottom_cate=bottom_cate_dict.get(answer)

                        if bottom_cate=='스커트' or bottom_cate=='숏팬츠':
                            bottom_fit_cate='null'
                        else:
                            # classify bottom fit
                            print(
                                ">>>>>BOTTOM fit: 0-와이드팬츠, 1-스키니진, 2-일자바지, 3-부츠컷")
                            # plt.imshow(resized_image)
                            resized_image = resized_image.reshape((1, 32, 32, 3))
                            predict = np.argmax(bottom_fit.predict(resized_image))
                            print("PREDICTION: ", predict)
                            user_input = int(
                                input(
                                    ">>>>>>>>If prediction is right, enter -1. If prediction is wrong, enter right class."))
                            if user_input == -1:
                                answer = predict
                            else:
                                answer = user_input
                            bottom_fit_cate = bottom_fit_dict.get(answer)

                        # Todo: classify color
                        print(">>>>>COLOR")
                        bottom_color= classify_color(file_directory)
                        print(bottom_color)

                    elif category[2]=='onepiece':
                        print("<This is ONE-PIECE>")

                        # classify one-piece sleeve length
                        print(
                            ">>>>>SLEEVE LENGTH: 0-긴팔, 1-민소매, 2-반팔")
                        # plt.imshow(resized_image)
                        resized_image = resized_image.reshape((1, 32, 32, 3))
                        predict = np.argmax(sleeve_length.predict(resized_image))
                        print("PREDICTION: ", predict)
                        user_input = int(
                            input(">>>>>>>>If prediction is right, enter -1. If prediction is wrong, enter right class."))
                        if user_input == -1:
                            answer = predict
                        else:
                            answer = user_input

                        if answer==0:
                            onepiece_temperature=3
                        elif answer==1:
                            onepiece_temperature=1
                        elif answer==2:
                            onepiece_temperature=2

                        onepiece_sleeve_length=sleeve_length_dict.get(answer)

                        # classify bottom length
                        print(
                             ">>>>>BOTTOM length: 0-미니, 1-맥시")
                        # plt.imshow(resized_image)
                        resized_image = resized_image.reshape((1, 32, 32, 3))
                        predict = np.argmax(bottom_length.predict(resized_image))
                        print("PREDICTION: ", predict)
                        user_input = int(
                            input(
                                ">>>>>>>>If prediction is right, enter -1. If prediction is wrong, enter right class."))
                        if user_input == -1:
                               answer = predict
                        else:
                            answer = user_input
                        onepiece_length=bottom_length_dict.get(answer)


                        # Todo: classify color
                        print(">>>>>COLOR")
                        onepiece_color = classify_color(file_directory)
                        print(onepiece_color)


        info_table_name='R_'+style_name+'_info'

        if outer_temperature!=0: # 아우터가 있는 스타일링인 경우
            if top_temperature==3:
                temperature=outer_temperature
            elif top_temperature==2:
                temperature=outer_temperature-1
            elif top_temperature==1:
                temperature=outer_temperature-2
        else: # 아우터가 없는 스타일링일 경우
            if top_temperature!=0: # 상의가 있는 경우
                temperature=top_temperature
            else: # 상의가 없는 경우 == 원피스인경우
                temperature=onepiece_temperature
        print("[TEMPERATURE] -> ",temperature)
        values=(ID,outer_cate,outer_color,outer_link,top_cate,top_sleeve_length,top_color,print_cate,top_link,bottom_cate,bottom_color,bottom_length_cate,bottom_link,onepiece_sleeve_length,onepiece_length,onepiece_color,onepiece_link,temperature,gender)

        info_sql='insert into '+info_table_name+'('+INFO_COLUMNS_NAME+') values (%s, %s, %s, %s, %s,%s, %s, %s, %s,%s, %s, %s, %s,%s, %s, %s, %s,%s, %s)'
        curs.execute(info_sql,values)
        db.commit()
        print("==============================================================================================")


