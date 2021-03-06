import boto3
from botocore.client import Config
from keras.models import load_model
from PIL import Image
import numpy as np
import color
from sklearn.cluster import KMeans
import cv2
import pymysql

BUCKET_NAME= 'smartmirror-bucket'
ACCESS_KEY_ID='AKIA462PQS3AXQ5D2BNE'
ACCESS_SECRET_KEY='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'

# def upload_img(file):
#     data=open(file,"rb")
#     s3=boto3.resource('s3',aws_access_key_id=ACCESS_KEY_ID,
#                       aws_secret_access_key=ACCESS_SECRET_KEY,
#                       config=Config(signature_version='s3v4'))
#     s3.Bucket(BUCKET_NAME).put_object(Key='user_image/'+file,Body=data,ContentType='image/jpg')
#
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

file='긴팔2.jpg'
file_cate='top'

# s3 client 생성
s3=boto3.client('s3')
# 파일 업로드
s3.upload_file(file,BUCKET_NAME,'user_image/'+file)

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
ID=1 # todo: change ID dynamically

image_table_name="U_"+file_cate+'_image'
info_table_name="U_"+file_cate+'_info'

# insert image to db
image_url='https://smartmirror-bucket.s3.ap-northeast-2.amazonaws.com/user_image/'+file
values=(ID,image_url)
image_sql='insert into '+image_table_name+'('+IMAGE_COLUMNS_NAME+') values (%s,%s)'
curs.execute(image_sql,values)
db.commit()

image=Image.open(file).convert('RGB')
image.show()
resized_image = image.resize((32, 32))
resized_image=np.array(resized_image)
resized_image=resized_image.astype('float32')/255.

if file_cate=='top':
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

elif file_cate=='bottom':
    INFO_COLUMNS_NAME='ID,category, length, color, temperature_section'

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
        bottom_temperature = 2
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
    values = (ID, bottom_cate, bottom_length, bottom_color, bottom_temperature)
    info_sql = 'insert into ' + info_table_name + '(' + INFO_COLUMNS_NAME + ') values (%s,%s,%s,%s,%s)'
    curs.execute(info_sql, values)
    db.commit()

elif file_cate=="onepiece":
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

elif file_cate=="outer":
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