<<<<<<< HEAD
# for user's clothes

import boto3
from botocore.client import Config
from keras.models import load_model
from PIL import Image
import numpy as np
import color
from sklearn.cluster import KMeans
import cv2
import pymysql
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
from keras.models import load_model


# 추가 노이즈 제거
def filterNoise(edgeImg):
    # Get rid of salt & pepper noise.
    count = 0
    lastMedian = edgeImg
    median = cv2.medianBlur(edgeImg, 3)
    while not np.array_equal(lastMedian, median):
        # get those pixels that gets zeroed out
        zeroed = np.invert(np.logical_and(median, edgeImg))
        edgeImg[zeroed] = 0

        count = count + 1
        if count > 50:
            break
        lastMedian = median
        median = cv2.medianBlur(edgeImg, 3)

# 가장 큰 윤곽감지
def findLargestContour(edgeImg):
        contours, hierarchy = cv2.findContours(
            edgeImg,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # From among them, find the contours with large surface area.
        contoursWithArea = []
        for contour in contours:
            area = cv2.contourArea(contour)
            contoursWithArea.append([contour, area])

        contoursWithArea.sort(key=lambda tupl: tupl[1], reverse=True)
        largestContour = contoursWithArea[0][0]
        return largestContour


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

def upload_user_image(file_name,category,ID):
    ###################################################################################3
    # remove background
    # 약간의 노이즈 제거(가우시안 블러)
    src = cv2.imread(file_name, 1)
    cv2.imshow('image',src)
    blurred = cv2.GaussianBlur(src, (5, 5), 0)

    blurred_float = blurred.astype(np.float32) / 255.0
    edgeDetector = cv2.ximgproc.createStructuredEdgeDetection("model.yml")
    edges = edgeDetector.detectEdges(blurred_float) * 255.0
    # cv2.imwrite('edge-raw.jpg', edges)

    # #추가 노이즈 제거
    # def filterNoise(edgeImg):
    #     # Get rid of salt & pepper noise.
    #     count = 0
    #     lastMedian = edgeImg
    #     median = cv2.medianBlur(edgeImg, 3)
    #     while not np.array_equal(lastMedian, median):
    #         # get those pixels that gets zeroed out
    #         zeroed = np.invert(np.logical_and(median, edgeImg))
    #         edgeImg[zeroed] = 0
    #
    #         count = count + 1
    #         if count > 50:
    #             break
    #         lastMedian = median
    #         median = cv2.medianBlur(edgeImg, 3)

    edges_8u = np.asarray(edges, np.uint8)
    filterNoise(edges_8u)
    # cv2.imwrite('edge.jpg', edges_8u)

    # #가장 큰 윤곽감지
    # def findLargestContour(edgeImg):
    #     contours, hierarchy = cv2.findContours(
    #         edgeImg,
    #         cv2.RETR_EXTERNAL,
    #         cv2.CHAIN_APPROX_SIMPLE
    #     )
    #
    #     # From among them, find the contours with large surface area.
    #     contoursWithArea = []
    #     for contour in contours:
    #         area = cv2.contourArea(contour)
    #         contoursWithArea.append([contour, area])
    #
    #     contoursWithArea.sort(key=lambda tupl: tupl[1], reverse=True)
    #     largestContour = contoursWithArea[0][0]
    #     return largestContour


    contour = findLargestContour(edges_8u)
    # Draw the contour on the original image
    contourImg = np.copy(src)
    cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
    # cv2.imwrite('contour.jpg', contourImg)

    mask = np.zeros_like(edges_8u)
    cv2.fillPoly(mask, [contour], 255)

    # calculate sure foreground area by dilating the mask
    mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)

    # mark inital mask as "probably background"
    # and mapFg as sure foreground
    trimap = np.copy(mask)
    trimap[mask == 0] = cv2.GC_BGD
    trimap[mask == 255] = cv2.GC_PR_BGD
    trimap[mapFg == 255] = cv2.GC_FGD

    # visualize trimap
    trimap_print = np.copy(trimap)
    trimap_print[trimap_print == cv2.GC_PR_BGD] = 128
    trimap_print[trimap_print == cv2.GC_FGD] = 255
    # cv2.imwrite('trimap.png', trimap_print)

    # run grabcut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)
    cv2.grabCut(src, trimap, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # create mask again
    mask2 = np.where(
        (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD),
        255,
        0
    ).astype('uint8')
    # cv2.imwrite('mask2.jpg', mask2)

    contour2 = findLargestContour(mask2)
    mask3 = np.zeros_like(mask2)
    cv2.fillPoly(mask3, [contour2], 255)

    # blended alpha cut-out
    mask3 = np.repeat(mask3[:, :, np.newaxis], 3, axis=2)
    mask4 = cv2.GaussianBlur(mask3, (3, 3), 0)
    alpha = mask4.astype(float) * 1.1  # making blend stronger
    alpha[mask3 > 0] = 255.0
    alpha[alpha > 255] = 255.0

    foreground = np.copy(src).astype(float)
    foreground[mask4 == 0] = 0
    background = np.ones_like(foreground, dtype=float) * 255.0

    # cv2.imwrite('foreground.png', foreground)
    # cv2.imwrite('background.png', background)
    # cv2.imwrite('alpha.png', alpha)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha / 255.0
    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)
    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)
    # Add the masked foreground and background.
    cutout = cv2.add(foreground, background)

    cv2.imwrite('result.jpg', cutout)


    ###################################################################################################
    # upload image to S3 and database
    BUCKET_NAME= 'smartmirror-bucket'
    ACCESS_KEY_ID='AKIA462PQS3AXQ5D2BNE'
    ACCESS_SECRET_KEY='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'

    file='result.jpg' # cutout



    # def upload_img(file):
    #     data=open(file,"rb")
    #     s3=boto3.resource('s3',aws_access_key_id=ACCESS_KEY_ID,
    #                       aws_secret_access_key=ACCESS_SECRET_KEY,
    #                       config=Config(signature_version='s3v4'))
    #     s3.Bucket(BUCKET_NAME).put_object(Key='user_image/'+file,Body=data,ContentType='image/jpg')

    # file='긴팔2.jpg'
    #
    # upload_img(file)

    # def upload_img(file):
    #     s3_client = boto3.client(
    #         's3',
    #         aws_access_key_id=ACCESS_KEY_ID,
    #         aws_secret_access_key=ACCESS_SECRET_KEY
    #     )
    #     response = s3_client.upload_file(
    #          f + '.jpg', BUCKET_NAME, 'user_image/'+category + f + '.jpg')
    # def classify_color(image_path):
    #     image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    #     try:
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         image = image.reshape((image.shape[0] * image.shape[1], 3))  # height, width 통합
    #
    #         k = 3
    #         clt = KMeans(n_clusters=k)
    #         clt.fit(image)
    #         hist = color.centroid_histogram(clt)
    #         back = 0
    #         for i in range(3):
    #             print(i, hist[i], color.find_color(clt.cluster_centers_[i]))
    #             if hist[i] < 0.1:
    #                 continue
    #             if color.find_color(clt.cluster_centers_[i]) == "흰색":
    #                 if back == 1:
    #                     print("흰색")
    #                     break
    #                 back = 1
    #             else:
    #                 answer_color = color.find_color(clt.cluster_centers_[i])
    #                 print(answer_color)
    #                 break
    #     except:
    #         answer_color='null'
    #
    #     return answer_color



    # s3 client 생성
    s3=boto3.client('s3')
    # 파일 업로드
    s3.upload_file(file,BUCKET_NAME,'user_image/'+file_name)

    # load cnn models
    bottom_categories=load_model('bottom_categories.h5')
    bottom_length=load_model('bottom_length.h5')
    outer_categories=load_model('outer_categories.h5')
    top_categories=load_model('top_categories.h5')
    sleeve_length=load_model('sleeve_length.h5')
    bottom_fit=load_model('bottom_fit.h5')
    print_=load_model('print.h5')

    # dictionary about classifying labels
    outer_cate_dict={0:'블레이저', 1:'트레이닝', 2:'무스탕',3:'트렌치코트',4:'코트',5:'트러커 자켓', 6:'라이더', 7: '블루종',8:'롱패딩', 9:'숏패딩',10:'가디건',11:'후드집업'}
    top_cate_dict={0:'맨투맨', 1:'민소매', 2:'반팔', 3:'셔츠', 4:'후드티'}
    sleeve_length_dict={0:'긴팔', 1:'민소매', 2:'반팔'}
    bottom_cate_dict={ 0:'스커트', 1:'레깅스', 2:'숏팬츠', 3:'슬랙스', 4:'조거팬츠'}
    bottom_length_dict={0:'미니',1:'맥시'}
    bottom_fit_dict={0:'와이드팬츠', 1:'스키니진', 2:'일자바지', 3:'부츠컷'}
    print_dict={0: '무지',1:'스트라이프',2:'그래픽'}

    outer_cate = 'null'
    outer_color = 'null'
    top_cate = 'null'
    top_sleeve_length = 'null'
    top_color = 'null'
    print_cate = 'null'
    bottom_cate = 'null'
    bottom_color = 'null'
    bottom_length_cate = 'null'
    onepiece_length = 'null'
    onepiece_sleeve_length = 'null'
    onepiece_color = 'null'
    outer_temperature=0
    top_temperature=0
    bottom_temperature=0
    onepiece_temperature=0

    IMAGE_COLUMNS_NAME='ID, image'

    db=pymysql.connect(host="52.79.59.24", user="minseo",password="minseopw", db="SmartMirror",charset='utf8')
    curs=db.cursor()




    data=open(file,"rb")
    #ID=1 # todo: change ID dynamically

    image_table_name="U_"+category+'_image'
    info_table_name="U_"+category+'_info'

    # insert image to db
    image_url='https://smartmirror-bucket.s3.ap-northeast-2.amazonaws.com/user_image/'+file_name
    values=(ID,image_url)
    image_sql='insert into '+image_table_name+'('+IMAGE_COLUMNS_NAME+') values (%s,%s)'
    curs.execute(image_sql,values)
    db.commit()

    image=Image.open(file).convert('RGB')
    image.show()
    resized_image = image.resize((32, 32))
    resized_image=np.array(resized_image)
    resized_image=resized_image.astype('float32')/255.

    if category=='top':
        INFO_COLUMNS_NAME='ID, category, sleevelength, color, print, temperature_section'
        # top category
        image_top_category=image.resize((64,64))
        image_top_category = np.array(image_top_category)
        image_top_category = image_top_category.astype('float32') / 255.

        print(
            ">>>>>TOP categories: 0-맨투맨, 1-민소매, 2-반팔, 3-셔츠, 4-후드티")
        image_top_category = image_top_category.reshape((1, 64, 64, 3))

        predict = np.argmax(top_categories.predict(image_top_category))
        print("PREDICTION: ", predict)

        if predict == 1:
            top_temperature = 1
        elif predict == 2:
            top_temperature = 2
        elif predict == 0 or predict == 4 or predict == 3:
            top_temperature = 3

        top_cate = top_cate_dict.get(predict)

        # top sleeve length
        print(
            ">>>>>SLEEVE LENGTH: 0-긴팔, 1-민소매, 2-반팔")
        # plt.imshow(resized_image)
        resized_image = resized_image.reshape((1, 32, 32, 3))
        predict = np.argmax(sleeve_length.predict(resized_image))
        print("PREDICTION: ", predict)

        top_sleeve_length = sleeve_length_dict.get(predict)

        # top print
        image_top_categories = image.resize((64, 64))
        image_top_categories = np.array(image_top_categories)
        image_top_categories = image_top_categories.astype('float32') / 255.
        print(
            ">>>>>PRINT: 0-무지, 1-스트라이프, 2-그래픽")
        # plt.imshow(resized_image)
        image_top_categories = image_top_categories.reshape((1, 64, 64, 3))
        predict = np.argmax(print_.predict(image_top_categories))
        print("PREDICTION: ", predict)

        print_cate = print_dict.get(predict)

        # color
        top_color = classify_color(file)
        print("color: ",top_color)

        # insert information to db
        values=(ID,top_cate,top_sleeve_length,top_color,print_cate,top_temperature)
        info_sql='insert into '+info_table_name+'('+INFO_COLUMNS_NAME+') values (%s,%s,%s,%s,%s,%s)'
        curs.execute(info_sql, values)
        db.commit()

    elif category=='bottom':
        INFO_COLUMNS_NAME='ID,category, length, color, temperature_section,bottom_fit'

        # bottom length
        print(
            ">>>>>BOTTOM length: 0-미니, 1-맥시")
        # plt.imshow(resized_image)
        resized_image = resized_image.reshape((1, 32, 32, 3))
        predict = np.argmax(bottom_length.predict(resized_image))
        print("PREDICTION: ", predict)

        if predict == 0:
            bottom_temperature = 1
        elif predict == 1:
            bottom_temperature = 3
        bottom_length_cate = bottom_length_dict.get(predict)

        # bottom category
        print(
            ">>>>>BOTTOM categories: 0-스커트, 1-레깅스, 2-숏팬츠, 3-슬랙스, 4-조거팬츠")
        # plt.imshow(resized_image)
        resized_image = resized_image.reshape((1, 32, 32, 3))
        predict = np.argmax(bottom_categories.predict(resized_image))
        print("PREDICTION: ", predict)

        bottom_cate = bottom_cate_dict.get(predict)

        if bottom_cate == '스커트' or bottom_cate == '숏팬츠':
            bottom_fit_cate = 'null'
        else:
            # bottom fit
            print(
                ">>>>>BOTTOM fit: 0-와이드팬츠, 1-스키니진, 2-일자바지, 3-부츠컷")
            # plt.imshow(resized_image)
            resized_image = resized_image.reshape((1, 32, 32, 3))
            predict = np.argmax(bottom_fit.predict(resized_image))
            print("PREDICTION: ", predict)

            bottom_fit_cate = bottom_fit_dict.get(predict)

        # color
        bottom_color = classify_color(file)
        print("color: ",bottom_color)

        # insert information to db
        values = (ID, bottom_cate, bottom_length, bottom_color, bottom_temperature,bottom_fit_cate)
        info_sql = 'insert into ' + info_table_name + '(' + INFO_COLUMNS_NAME + ') values (%s,%s,%s,%s,%s,%s)'
        curs.execute(info_sql, values)
        db.commit()

    elif category=="onepiece":
        INFO_COLUMNS_NAME='ID, sleevelength, length, color, temperature_section'
        # sleeve length
        print(
            ">>>>>SLEEVE LENGTH: 0-긴팔, 1-민소매, 2-반팔")
        # plt.imshow(resized_image)
        resized_image = resized_image.reshape((1, 32, 32, 3))
        predict = np.argmax(sleeve_length.predict(resized_image))
        print("PREDICTION: ", predict)

        if predict == 0:
            onepiece_temperature = 3
        elif predict == 1:
            onepiece_temperature = 1
        elif predict == 2:
            onepiece_temperature = 2

        onepiece_sleeve_length = sleeve_length_dict.get(predict)

        # length
        print(
            ">>>>>length: 0-미니, 1-맥시")
        # plt.imshow(resized_image)
        resized_image = resized_image.reshape((1, 32, 32, 3))
        predict = np.argmax(bottom_length.predict(resized_image))
        print("PREDICTION: ", predict)

        # color
        onepiece_color = classify_color(file)
        print("color: ",onepiece_color)

        # insert information to db
        values = (ID, onepiece_sleeve_length, onepiece_length, onepiece_color, onepiece_temperature)
        info_sql = 'insert into ' + info_table_name + '(' + INFO_COLUMNS_NAME + ') values (%s,%s,%s,%s,%s)'
        curs.execute(info_sql, values)
        db.commit()

    elif category=="outer":
        INFO_COLUMNS_NAME='ID, category, color, temperature_section'

        # category
        print(
            ">>>>>OUTER categories: 0-블레이저, 1-트레이닝, 2-무스탕, 3-트렌치코트, 4-코트, 5-트러커 자켓, 6-라이더, 7- 블루종, 8-롱패딩, 9-숏패딩, 10-가디건, 11-후드집업")
        # plt.imshow(resized_image)
        resized_image = resized_image.reshape((1, 32, 32, 3))
        predict = np.argmax(outer_categories.predict(resized_image))
        print("PREDICTION: ", predict)

        if predict == 1 or predict == 11 or predict == 3:
            outer_temperature = 4
        elif predict == 0 or predict == 5 or predict == 6 or predict == 7 or predict == 11:
            outer_temperature = 5
        elif predict == 2 or predict == 4 or predict == 9:
            outer_temperature = 6
        elif predict == 8:
            outer_temperature = 7

        outer_cate = outer_cate_dict.get(predict)

        # color
        outer_color = classify_color(file)
        print("color: ", outer_color)

        # insert information to db
        values = (ID, outer_cate, outer_color, outer_temperature)
        info_sql = 'insert into ' + info_table_name + '(' + INFO_COLUMNS_NAME + ') values (%s,%s,%s,%s)'
        curs.execute(info_sql, values)
        db.commit()







# file_name="outer_6.jpg"
# file=file_name.replace('.','_')
# print(file_name)
# split=file.split('_')
# category=split[0]
# id=split[1]
#
# upload_user_image(file_name,category,id)
=======
# -*- coding: utf-8 -*-

# for user's clothes

import boto3
import color
from sklearn.cluster import KMeans
import cv2
import pymysql
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model



def filterNoise(edgeImg):
    # Get rid of salt & pepper noise.
    count = 0
    lastMedian = edgeImg
    median = cv2.medianBlur(edgeImg, 3)
    while not np.array_equal(lastMedian, median):
        # get those pixels that gets zeroed out
        zeroed = np.invert(np.logical_and(median, edgeImg))
        edgeImg[zeroed] = 0

        count = count + 1
        if count > 50:
            break
        lastMedian = median
        median = cv2.medianBlur(edgeImg, 3)


def findLargestContour(edgeImg):
        contours, hierarchy = cv2.findContours(
            edgeImg,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # From among them, find the contours with large surface area.
        contoursWithArea = []
        for contour in contours:
            area = cv2.contourArea(contour)
            contoursWithArea.append([contour, area])

        contoursWithArea.sort(key=lambda tupl: tupl[1], reverse=True)
        largestContour = contoursWithArea[0][0]
        return largestContour


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
                if color.find_color(clt.cluster_centers_[i]) == "white":
                    if back == 1:
                        print("white")
                        break
                    back = 1
                else:
                    answer_color = color.find_color(clt.cluster_centers_[i])
                    print(answer_color)
                    break
        except:
            answer_color='null'

        return answer_color

def upload_user_image(file_name,category,ID):
    # remove background
    src = cv2.imread(file_name, 1)
    blurred = cv2.GaussianBlur(src, (5, 5), 0)

    blurred_float = blurred.astype(np.float32) / 255.0
    edgeDetector = cv2.ximgproc.createStructuredEdgeDetection("model.yml")
    edges = edgeDetector.detectEdges(blurred_float) * 255.0
    edges_8u = np.asarray(edges, np.uint8)
    filterNoise(edges_8u)
    contour = findLargestContour(edges_8u)
    # Draw the contour on the original image
    contourImg = np.copy(src)
    cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
    mask = np.zeros_like(edges_8u)
    cv2.fillPoly(mask, [contour], 255)
    # calculate sure foreground area by dilating the mask
    mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)
    # mark inital mask as "probably background"
    # and mapFg as sure foreground
    trimap = np.copy(mask)
    trimap[mask == 0] = cv2.GC_BGD
    trimap[mask == 255] = cv2.GC_PR_BGD
    trimap[mapFg == 255] = cv2.GC_FGD
    # visualize trimap
    trimap_print = np.copy(trimap)
    trimap_print[trimap_print == cv2.GC_PR_BGD] = 128
    trimap_print[trimap_print == cv2.GC_FGD] = 255
     # run grabcut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)
    cv2.grabCut(src, trimap, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    # create mask again
    mask2 = np.where(
        (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD),
        255,
        0
    ).astype('uint8')
    contour2 = findLargestContour(mask2)
    mask3 = np.zeros_like(mask2)
    cv2.fillPoly(mask3, [contour2], 255)
    # blended alpha cut-out
    mask3 = np.repeat(mask3[:, :, np.newaxis], 3, axis=2)
    mask4 = cv2.GaussianBlur(mask3, (3, 3), 0)
    alpha = mask4.astype(float) * 1.1  # making blend stronger
    alpha[mask3 > 0] = 255.0
    alpha[alpha > 255] = 255.0
    foreground = np.copy(src).astype(float)
    foreground[mask4 == 0] = 0
    background = np.ones_like(foreground, dtype=float) * 255.0
    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha / 255.0
    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)
    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)
    # Add the masked foreground and background.
    cutout = cv2.add(foreground, background)

    cv2.imwrite('result.jpg', cutout)

    # upload image to S3 and database
    BUCKET_NAME= 'smartmirror-bucket'
    ACCESS_KEY_ID='AKIA462PQS3AXQ5D2BNE'
    ACCESS_SECRET_KEY='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'

    file='result.jpg' # cutout
    s3 = boto3.client('s3', region_name="ap-northeast-2", endpoint_url="ec2.ap-northeast-2.amazonaws.com", aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=ACCESS_SECRET_KEY)

    # 파일 업로드
    s3.upload_file(file,BUCKET_NAME,'user_image/'+file_name)

    # load cnn models
    bottom_categories=load_model('bottom_categories.h5')
    bottom_length=load_model('bottom_length.h5')
    outer_categories=load_model('outer_categories.h5')
    top_categories=load_model('top_categories.h5')
    sleeve_length=load_model('sleeve_length.h5')
    bottom_fit=load_model('bottom_fit.h5')
    print_=load_model('print.h5')
     
    # dictionary about classifying labels
    outer_cate_dict={0:'블레이저', 1:'트레이닝', 2:'무스탕',3:'트렌치코트',4:'코트',5:'트러커 자켓', 6:'라이더', 7: '블루종',8:'롱패딩', 9:'숏패딩',10:'가디건',11:'후드집업'}
    top_cate_dict={0:'맨투맨', 1:'민소매', 2:'반팔', 3:'셔츠', 4:'후드티'}
    sleeve_length_dict={0:'긴팔', 1:'민소매', 2:'반팔'}
    bottom_cate_dict={ 0:'스커트', 1:'레깅스', 2:'숏팬츠', 3:'슬랙스', 4:'조거팬츠'}
    bottom_length_dict={0:'미니',1:'맥시'}
    bottom_fit_dict={0:'와이드팬츠', 1:'스키니진', 2:'일자바지', 3:'부츠컷'}
    print_dict={0: '무지',1:'스트라이프',2:'그래픽'}

    outer_cate = 'null'
    outer_color = 'null'
    top_cate = 'null'
    top_sleeve_length = 'null'
    top_color = 'null'
    print_cate = 'null'
    bottom_cate = 'null'
    bottom_color = 'null'
    bottom_length_cate = 'null'
    onepiece_length = 'null'
    onepiece_sleeve_length = 'null'
    onepiece_color = 'null'
    outer_temperature=0
    top_temperature=0
    bottom_temperature=0
    onepiece_temperature=0

    IMAGE_COLUMNS_NAME='ID, image'

    db=pymysql.connect(host="54.180.67.155", user="minseo", password="minseopw", db="SmartMirror",charset='utf8')
    curs=db.cursor()

    data=open(file,"rb")
    #ID=1 # todo: change ID dynamically

    image_table_name="U_"+category+'_image'
    info_table_name="U_"+category+'_info'

    # insert image to db
    image_url='https://smartmirror-bucket.s3.ap-northeast-2.amazonaws.com/user_image/'+file_name
    values=(ID,image_url)
    image_sql='insert into '+image_table_name+'('+IMAGE_COLUMNS_NAME+') values (%s,%s)'
    curs.execute(image_sql,values)
    db.commit()

    image=Image.open(file).convert('RGB')
    image.show()
    resized_image = image.resize((32, 32))
    resized_image=np.array(resized_image)
    resized_image=resized_image.astype('float32')/255.

    if category == 'top':
        INFO_COLUMNS_NAME ='ID, category, sleevelength, color, print, temperature_section'
        # top category
        image_top_category=image.resize((64,64))
        image_top_category = np.array(image_top_category)
        image_top_category = image_top_category.astype('float32') / 255.

        print(">>>>>TOP categories: 0-맨투맨, 1-민소매, 2-반팔, 3-셔츠, 4-후드티")
        image_top_category = image_top_category.reshape((1, 64, 64, 3))

        predict = np.argmax(top_categories.predict(image_top_category))
        print("PREDICTION: ", predict)

        if predict == 1:
            top_temperature = 1
        elif predict == 2:
            top_temperature = 2
        elif predict == 0 or predict == 4 or predict == 3:
            top_temperature = 3

        top_cate = top_cate_dict.get(predict)

        # top sleeve length
        print(">>>>>SLEEVE LENGTH: 0-긴팔, 1-민소매, 2-반팔")
        resized_image = resized_image.reshape((1, 32, 32, 3))
        predict = np.argmax(sleeve_length.predict(resized_image))
        print("PREDICTION: ", predict)

        top_sleeve_length = sleeve_length_dict.get(predict)

        # top print
        image_top_categories = image.resize((64, 64))
        image_top_categories = np.array(image_top_categories)
        image_top_categories = image_top_categories.astype('float32') / 255.
        print(">>>>>PRINT: 0-무지, 1-스트라이프, 2-그래픽")
        image_top_categories = image_top_categories.reshape((1, 64, 64, 3))
        predict = np.argmax(print_.predict(image_top_categories))
        print("PREDICTION: ", predict)

        print_cate = print_dict.get(predict)

        # color
        top_color = classify_color(file)
        print("color: ",top_color)

        # insert information to db
        values=(ID,top_cate,top_sleeve_length,top_color,print_cate,top_temperature)
        info_sql='insert into '+info_table_name+'('+INFO_COLUMNS_NAME+') values (%s,%s,%s,%s,%s,%s)'
        curs.execute(info_sql, values)
        db.commit()

    elif category == 'bottom':
        INFO_COLUMNS_NAME='ID,category, length, color, temperature_section,bottom_fit'

        # bottom length
        print(">>>>>BOTTOM length: 0-미니, 1-맥시")
        resized_image = resized_image.reshape((1, 32, 32, 3))
        predict = np.argmax(bottom_length.predict(resized_image))
        print("PREDICTION: ", predict)

        if predict == 0:
            bottom_temperature = 1
        elif predict == 1:
            bottom_temperature = 3
        bottom_length_cate = bottom_length_dict.get(predict)

        # bottom category
        print(">>>>>BOTTOM categories: 0-스커트, 1-레깅스, 2-숏팬츠, 3-슬랙스, 4-조거팬츠")
        resized_image = resized_image.reshape((1, 32, 32, 3))
        predict = np.argmax(bottom_categories.predict(resized_image))
        print("PREDICTION: ", predict)

        bottom_cate = bottom_cate_dict.get(predict)

        if bottom_cate == '스커트' or bottom_cate == '숏팬츠':
            bottom_fit_cate = 'null'
        else:
            # bottom fit
            print(">>>>>BOTTOM fit: 0-와이드팬츠, 1-스키니진, 2-일자바지, 3-부츠컷")
            resized_image = resized_image.reshape((1, 32, 32, 3))
            predict = np.argmax(bottom_fit.predict(resized_image))
            print("PREDICTION: ", predict)

            bottom_fit_cate = bottom_fit_dict.get(predict)

        # color
        bottom_color = classify_color(file)
        print("color: ",bottom_color)

        # insert information to db
        values = (ID, bottom_cate, bottom_length, bottom_color, bottom_temperature,bottom_fit_cate)
        info_sql = 'insert into ' + info_table_name + '(' + INFO_COLUMNS_NAME + ') values (%s,%s,%s,%s,%s,%s)'
        curs.execute(info_sql, values)
        db.commit()

    elif category == "onepiece":
        INFO_COLUMNS_NAME='ID, sleevelength, length, color, temperature_section'
        # sleeve length
        print(">>>>>SLEEVE LENGTH: 0-긴팔, 1-민소매, 2-반팔")
        resized_image = resized_image.reshape((1, 32, 32, 3))
        predict = np.argmax(sleeve_length.predict(resized_image))
        print("PREDICTION: ", predict)

        if predict == 0:
            onepiece_temperature = 3
        elif predict == 1:
            onepiece_temperature = 1
        elif predict == 2:
            onepiece_temperature = 2

        onepiece_sleeve_length = sleeve_length_dict.get(predict)

        # length
        print(">>>>>length: 0-미니, 1-맥시")
        resized_image = resized_image.reshape((1, 32, 32, 3))
        predict = np.argmax(bottom_length.predict(resized_image))
        print("PREDICTION: ", predict)

        # color
        onepiece_color = classify_color(file)
        print("color: ",onepiece_color)

        # insert information to db
        values = (ID, onepiece_sleeve_length, onepiece_length, onepiece_color, onepiece_temperature)
        info_sql = 'insert into ' + info_table_name + '(' + INFO_COLUMNS_NAME + ') values (%s,%s,%s,%s,%s)'
        curs.execute(info_sql, values)
        db.commit()

    elif category == "outer":
        INFO_COLUMNS_NAME='ID, category, color, temperature_section'

        # category
        print(">>>>>OUTER categories: 0-블레이저, 1-트레이닝, 2-무스탕, 3-트렌치코트, 4-코트, 5-트러커 자켓, 6-라이더, 7- 블루종, 8-롱패딩, 9-숏패딩, 10-가디건, 11-후드집업")
        resized_image = resized_image.reshape((1, 32, 32, 3))
        predict = np.argmax(outer_categories.predict(resized_image))
        print("PREDICTION: ", predict)

        if predict == 1 or predict == 11 or predict == 3:
            outer_temperature = 4
        elif predict == 0 or predict == 5 or predict == 6 or predict == 7 or predict == 11:
            outer_temperature = 5
        elif predict == 2 or predict == 4 or predict == 9:
            outer_temperature = 6
        elif predict == 8:
            outer_temperature = 7

        outer_cate = outer_cate_dict.get(predict)

        # color
        outer_color = classify_color(file)
        print("color: ", outer_color)

        # insert information to db
        values = (ID, outer_cate, outer_color, outer_temperature)
        info_sql = 'insert into ' + info_table_name + '(' + INFO_COLUMNS_NAME + ') values (%s,%s,%s,%s)'
        curs.execute(info_sql, values)
        db.commit()
>>>>>>> 717f58a4f915212e1b92bf7a6c3785e436c3d4d0
