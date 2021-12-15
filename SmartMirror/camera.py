# To upload user clothes into database
# detect upper body and lower body
# 상의 하의 좌표에 맞게 이미지 자르고 데이터베이스에 저장

import cv2
import upload_s3
import file_id_info

cap = cv2.VideoCapture(0) #return 0 or -1

upper_path = 'Downloads/haarcascade_upperbody.xml'
lower_path = 'Downloads/haarcascade_lowerbody.xml'
upperCascade = cv2.CascadeClassifier(upper_path)
lowerCascade = cv2.CascadeClassifier(lower_path)
result = []


def capture():
    while cap.isOpened():
        ret, img = cap.read()

        out = {}

        if cv2.waitKey(1)&0xFF == ord('q'):
            break
        if not ret:
            print('no camera connected!')

        else:
            imageGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            try:
                lowerRect = lowerCascade.detectMultiScale(imageGray, scaleFactor=1.3, minNeighbors=1, minSize=(1,1))
                temp = []
                temp2 = []
                for x,y,w,h in lowerRect:
                    if w>100:
                        print("lower: %d, %d, %d, %d"%(x,y,w,h))
                        temp = [x,y,w,h]
                        roi_color = img[y:y + h, x:x + w]
                        upperRect = upperCascade.detectMultiScale(imageGray, scaleFactor=1.3, minNeighbors=1, minSize=(1, 1))
                        for lx,ly,lw,lh in upperRect:
                            if lw>100 and lx<350:
                                print("upper: %d, %d, %d, %d"%(lx,ly,lw,lh))
                                ly = 0
                                roi_color_u = img[ly:ly + lh, lx:lx + lw]
                                temp2=[lx,ly,lw,lh]
                if temp and temp2:
                    result = [temp,temp2]
                    cap.release()
                    cv2.destroyAllWindows()
                    ######################################## bottom
                    id = file_id_info.get_id('bottom')
                    bottom_file_name = "bottom_" + str(id) + '.jpg'
                    cv2.imwrite(bottom_file_name, roi_color)
                    file_id_info.set_id('bottom', id + 1)

                    # upload user's cloth image into database and s3 storage
                    upload_s3.upload_user_image(bottom_file_name, "bottom", file_id_info.get_id("bottom"))

                    ######################################## top
                    id = file_id_info.get_id('top')
                    top_file_name = "top_" + str(id) + '.jpg'
                    cv2.imwrite(top_file_name, roi_color_u)
                    file_id_info.set_id('top', id + 1)

                    # upload user's cloth image into database and s3 storage
                    upload_s3.upload_user_image(top_file_name, "top", file_id_info.get_id("top"))
                    return result
            except ValueError as e:
                print("ERROR: " + str(e))

            cv2.imshow('camera-0', img)
    cap.release()
    cv2.destroyAllWindows()
