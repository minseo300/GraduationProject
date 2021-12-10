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
import pandas as pd
import pickle

db=pymysql.connect(host="52.79.59.24", user="minseo",password="minseopw", db="SmartMirror",charset='utf8')
curs=db.cursor()

styling_table=["casual","chic","dandy","formal","girlish","sports","romantic","street"]

topCate_bottomCate={
                        '맨투맨':[0,0,0,0,0],
                        '민소매':[0,0,0,0,0],
                        '반팔':[0,0,0,0,0],
                        '셔츠':[0,0,0,0,0],
                        '후드티':[0,0,0,0,0]
                    }
topCate_bottomCate=pd.DataFrame(topCate_bottomCate,index=['스커트','레깅스','숏팬츠','슬랙스','조거팬츠'])
topColor_bottomColor={
                        'black':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'silver':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'gray':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'white':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'purple':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'red':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'pink':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'green':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'khaki':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'yellow':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'navy':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'blue':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'skyblue':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'beige':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'mint':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'brown':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'orange':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'gold':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'lavender':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'silver':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                    }
topColor_bottomColor=pd.DataFrame(topColor_bottomColor,index=['black','silver','gray','white','purple', 'red', 'pink', 'green', 'khaki', 'yellow', 'navy', 'blue', 'skyblue', 'beige', 'mint', 'brown', 'orange', 'gold', 'lavender', 'silver'])
topCate_bottomFit={
                        '맨투맨':[0,0,0,0],
                        '민소매':[0,0,0,0],
                        '반팔':[0,0,0,0],
                        '셔츠':[0,0,0,0],
                        '후드티':[0,0,0,0]
                    }
topCate_bottomFit=pd.DataFrame(topCate_bottomFit,index=['와이드팬츠','스키니진','일자바지','부츠컷'])
topCate_outerCate={
                        '맨투맨':[0,0,0,0,0,0,0,0,0,0,0,0],
                        '민소매':[0,0,0,0,0,0,0,0,0,0,0,0],
                        '반팔':[0,0,0,0,0,0,0,0,0,0,0,0],
                        '셔츠':[0,0,0,0,0,0,0,0,0,0,0,0],
                        '후드티':[0,0,0,0,0,0,0,0,0,0,0,0]
                    }
topCate_outerCate=pd.DataFrame(topCate_outerCate,index=['블레이저','트레이닝','무스탕','트렌치코트','코트','트러커 자켓','라이더','블루종','롱패딩','숏패딩','가디건','후드집업'])
topColor_outerColor={
                        'black':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'silver':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'gray':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'white':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'purple':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'red':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'pink':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'green':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'khaki':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'yellow':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'navy':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'blue':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'skyblue':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'beige':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'mint':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'brown':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'orange':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'gold':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'lavender':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'silver':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

                    }
topColor_outerColor=pd.DataFrame(topColor_outerColor,index=['black','silver','gray','white','purple', 'red', 'pink', 'green', 'khaki', 'yellow', 'navy', 'blue', 'skyblue', 'beige', 'mint', 'brown', 'orange', 'gold', 'lavender', 'silver'])
outerColor_bottomColor={
                        'black':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'silver':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'gray':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'white':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'purple':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'red':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'pink':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'green':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'khaki':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'yellow':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'navy':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'blue':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'skyblue':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'beige':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'mint':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'brown':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'orange':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'gold':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'lavender':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        'silver':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                    }
outerColor_bottomColor=pd.DataFrame(outerColor_bottomColor,index=['black','silver','gray','white','purple', 'red', 'pink', 'green', 'khaki', 'yellow', 'navy', 'blue', 'skyblue', 'beige', 'mint', 'brown', 'orange', 'gold', 'lavender', 'silver'])
outerCate_bottomCate={
                        '블레이저':[0,0,0,0,0],
                        '트레이닝':[0,0,0,0,0],
                        '무스탕':[0,0,0,0,0],
                        '트렌치코트':[0,0,0,0,0],
                        '코트':[0,0,0,0,0],
                        '트러커 자켓':[0,0,0,0,0],
                        '라이더':[0,0,0,0,0],
                        '블루종':[0,0,0,0,0],
                        '롱패딩':[0,0,0,0,0],
                        '숏패딩':[0,0,0,0,0],
                        '가디건':[0,0,0,0,0],
                        '후드집업':[0,0,0,0,0]
                    }
outerCate_bottomCate=pd.DataFrame(outerCate_bottomCate,index=['스커트','레깅스','숏팬츠','슬랙스','조거팬츠'])
outerCate_bottomFit={
                        '블레이저': [0,0,0,0],
                        '트레이닝': [0,0,0,0],
                        '무스탕': [0,0,0,0],
                        '트렌치코트': [0,0,0,0],
                        '코트': [0,0,0,0],
                        '트러커 자켓': [0,0,0,0],
                        '라이더': [0,0,0,0],
                        '블루종': [0,0,0,0],
                        '롱패딩': [0,0,0,0],
                        '숏패딩': [0,0,0,0],
                        '가디건': [0,0,0,0],
                        '후드집업': [0,0,0,0]
                    }
outerCate_bottomFit=pd.DataFrame(outerCate_bottomFit,index=['와이드팬츠','스키니진','일자바지','부츠컷'])

table="for_matrix"
sql = "select outer_cate,outer_color,top_cate,top_color, bottom_cate, bottom_color, bottom_fit from " + table + ";"
curs.execute(sql)
result = curs.fetchall()
result = pd.DataFrame(result)
print(result)
print('----------')

for index, row in result.iterrows():
    print(row)
    outer_cate=row[0]
    outer_color = row[1]
    top_cate = row[2]
    top_color = row[3]
    bottom_cate = row[4]
    bottom_color = row[5]
    bottom_fit = row[6]

    try:
        topCate_bottomCate.loc[bottom_cate][top_cate]=topCate_bottomCate.loc[bottom_cate][top_cate]+1
        topColor_bottomColor.loc[bottom_color][top_color]=topColor_bottomColor.loc[bottom_color][top_color]+1
        topCate_outerCate.loc[outer_cate][top_cate]=topCate_outerCate.loc[outer_cate][top_cate]+1
        topColor_outerColor.loc[outer_color][top_color]=topColor_outerColor.loc[outer_color][top_color]+1
        outerColor_bottomColor.loc[bottom_color][outer_color]=outerColor_bottomColor.loc[bottom_color][outer_color]+1
        outerCate_bottomCate.loc[bottom_cate][outer_cate]=outerCate_bottomCate.loc[bottom_cate][outer_cate]+1
        if bottom_fit!="null":
            topCate_bottomFit.loc[bottom_fit][top_cate]=topCate_bottomFit.loc[bottom_fit][top_cate]+1
            outerCate_bottomFit.loc[bottom_fit][outer_cate]= outerCate_bottomFit.loc[bottom_fit][outer_cate]+1
    except:
        continue
    # else:
    #     topCate_bottomFit.loc[bottom_fit][top_cate]=0
    #     outerCate_bottomFit.loc[bottom_fit][outer_cate]=0

print(topCate_bottomCate)
topCate_bottomCate.to_pickle("topCate_bottomCate.pkl")
print(topColor_bottomColor)
topColor_bottomColor.to_pickle("topColor_bottomColor.pkl")
print(topCate_bottomFit)
topCate_bottomFit.to_pickle("topCate_bottomFit.pkl")
print(topCate_outerCate)
topCate_outerCate.to_pickle("topCate_outerCate.pkl")
print(topColor_outerColor)
topColor_outerColor.to_pickle("topColor_outerColor.pkl")
print(outerColor_bottomColor)
outerColor_bottomColor.to_pickle("outerColor_bottomColor.pkl")
print(outerCate_bottomCate)
outerCate_bottomCate.to_pickle("outerCate_bottomCate.pkl")
print(outerCate_bottomFit)
outerCate_bottomFit.to_pickle("outerCate_bottomFit.pkl")













# for styling in styling_table:
#     sql_table="R_"+styling+"_info"
#
#     # 상의 카테고리, 하의 카테고리
#     sql="select top_cate, bottom_cate from "+sql_table+";"
#     curs.execute(sql)
#     result=curs.fetchall()
#     result=pd.DataFrame(result)
#     print(result)
#     print('----------')
#     for index, row in result.iterrows():
#         top_category=row[0]
#         bottom_category=row[1]
#         print("top cate: ", top_category)
#         print("bottom cate: ", bottom_category)
#         if top_category!="null" and bottom_category!="null":
#             topCate_bottomCate.loc[bottom_category][top_category]=topCate_bottomCate.loc[bottom_category][top_category]+1
#     print("==================================================")
#     print(topCate_bottomCate)
#     topCate_bottomCate.to_pickle("topCate_bottomCate.pkl")
#
#     # 상의 색상, 하의 색상
#     sql = "select top_color, bottom_color from " + sql_table + ";"
#     curs.execute(sql)
#     result = curs.fetchall()
#     result = pd.DataFrame(result)
#     print(result)
#     print('----------')
#     for index, row in result.iterrows():
#         top_color = row[0]
#         bottom_color= row[1]
#         # print("top_color: ", top_color)
#         # print("bottom_color: ", bottom_color)
#         if top_color != "null" and bottom_color != "null":
#             topColor_bottomColor.loc[bottom_color][top_color] = topColor_bottomColor.loc[bottom_color][top_color] + 1
#     print("==================================================")
#     print(topColor_bottomColor)
#     topColor_bottomColor.to_pickle("topColor_bottomColor.pkl")
#
#     # 상의 카테고리, 하의 핏
#     sql = "select top_cate, bottom_fit from " + sql_table + ";"
#     curs.execute(sql)
#     result = curs.fetchall()
#     result = pd.DataFrame(result)
#     print(result)
#     print('----------')
#     for index, row in result.iterrows():
#         top_category = row[0]
#         bottom_fit= row[1]
#         # print("top_category: ", top_category)
#         # print("bottom_fit: ", bottom_fit)
#         if top_category != "null" and bottom_fit != "null":
#             topCate_bottomFit.loc[bottom_fit][top_category] = topCate_bottomFit.loc[bottom_fit][top_category] + 1
#     print("==================================================")
#     print(topCate_bottomFit)
#     topCate_bottomFit.to_pickle("topCate_bottomFit.pkl")
#
#     # 상의 카테고리, 아우터 카테고리
#     sql = "select top_cate, outer_cate from " + sql_table + ";"
#     curs.execute(sql)
#     result = curs.fetchall()
#     result = pd.DataFrame(result)
#     print(result)
#     print('----------')
#     for index, row in result.iterrows():
#         top_category = row[0]
#         outer_category = row[1]
#         # print("top_category: ", top_category)
#         # print("outer_category: ", outer_category)
#         if top_category != "null" and outer_category != "null":
#             topCate_outerCate.loc[outer_category][top_category] = topCate_outerCate.loc[outer_category][top_category] + 1
#     print("==================================================")
#     print(topCate_outerCate)
#     topCate_outerCate.to_pickle("topCate_outerCate.pkl")
#
#     # 상의 색상, 아우터 색상
#     sql = "select top_color, outer_color from " + sql_table + ";"
#     curs.execute(sql)
#     result = curs.fetchall()
#     result = pd.DataFrame(result)
#     print(result)
#     print('----------')
#     for index, row in result.iterrows():
#         top_color = row[0]
#         outer_color = row[1]
#         # print("top_color: ", top_color)
#         # print("outer_color: ", outer_color)
#         if top_color != "null" and outer_color != "null":
#             topColor_outerColor.loc[outer_color][top_color] = topColor_outerColor.loc[outer_color][top_color] + 1
#     print("==================================================")
#     print(topColor_outerColor)
#     topColor_outerColor.to_pickle("topColor_outerColor.pkl")
#
#     # 아우터 색상, 하의 색상
#     sql = "select outer_color, bottom_color from " + sql_table + ";"
#     curs.execute(sql)
#     result = curs.fetchall()
#     result = pd.DataFrame(result)
#     print(result)
#     print('----------')
#     for index, row in result.iterrows():
#         outer_color = row[0]
#         bottom_color = row[1]
#         # print("outer_color: ", outer_color)
#         # print("bottom_color: ", bottom_color)
#         if outer_color != "null" and bottom_color != "null":
#             outerColor_bottomColor.loc[bottom_color][outer_color] = outerColor_bottomColor.loc[bottom_color][outer_color] + 1
#     print("==================================================")
#     print(outerColor_bottomColor)
#     outerColor_bottomColor.to_pickle("outerColor_bottomColor.pkl")
#
#     # 아우터 카테고리, 하의 카테고리
#     sql = "select outer_cate, bottom_cate from " + sql_table + ";"
#     curs.execute(sql)
#     result = curs.fetchall()
#     result = pd.DataFrame(result)
#     print(result)
#     print('----------')
#     for index, row in result.iterrows():
#         outer_category = row[0]
#         bottom_category = row[1]
#         # print("outer_category: ", outer_category)
#         # print("bottom_category: ", bottom_category)
#         if outer_category != "null" and bottom_category != "null":
#             outerCate_bottomCate.loc[bottom_category][outer_category] = outerCate_bottomCate.loc[bottom_category][outer_category]+ 1
#     print("==================================================")
#     print(outerCate_bottomCate)
#     outerCate_bottomCate.to_pickle("outerCate_bottomCate.pkl")
#
#     # 아우터 카테고리, 하의 핏
#     sql = "select outer_cate, bottom_fit from " + sql_table + ";"
#     curs.execute(sql)
#     result = curs.fetchall()
#     result = pd.DataFrame(result)
#     print(result)
#     print('----------')
#     for index, row in result.iterrows():
#         outer_category = row[0]
#         bottom_fit = row[1]
#         # print("outer_category: ", outer_category)
#         # print("bottom_category: ", bottom_fit)
#         if outer_category != "null" and bottom_fit != "null":
#             outerCate_bottomFit.loc[bottom_fit][outer_category] = outerCate_bottomFit.loc[bottom_fit][outer_category] + 1
#     print("==================================================")
#     print(outerCate_bottomFit)
#     outerCate_bottomFit.to_pickle("outerCate_bottomFit.pkl")