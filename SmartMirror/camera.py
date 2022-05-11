# To upload user clothes into database
# detect upper body and lower body

import cv2
import upload_s3
import file_id_info

cap = cv2.VideoCapture(0) #return 0 or -1

upper_path = 'haarcascade_upperbody.xml'
lower_path = 'haarcascade_lowerbody.xml'
upperCascade = cv2.CascadeClassifier(upper_path)
lowerCascade = cv2.CascadeClassifier(lower_path)
result = []

#find user`s coordinate and return
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
                #find lower coordinate
                for x,y,w,h in lowerRect:
                    print("lower: %d, %d, %d, %d"%(x,y,w,h))
                    temp = [x,y,w,h]
                    roi_color = img[y:y + h, x:x + w]
                    upperRect = upperCascade.detectMultiScale(imageGray, scaleFactor=1.3, minNeighbors=1, minSize=(1, 1))
                    #find upper coordinate
                    for lx,ly,lw,lh in upperRect:
                        if lw>10:
                            print("upper: %d, %d, %d, %d"%(lx,ly,lw,lh))
                            ly = 0
                            roi_color_u = img[ly:ly + lh, lx:lx + lw]
                            temp2=[lx,ly,lw,lh]
                #if detect all coordinate
                if temp and temp2:
                    result = [temp,temp2]
                    cap.release()
                    cv2.destroyAllWindows()
                    # bottom
                    id = file_id_info.get_id('bottom')
                    print('id', id)
                    bottom_file_name = "bottom_" + str(id) + '.jpg'
                    cv2.imwrite(bottom_file_name, roi_color)
                    file_id_info.set_id('bottom', id + 1)
                    # upload user's bottom cloth image into database and s3 storage
                    upload_s3.upload_user_image(bottom_file_name, "bottom", file_id_info.get_id("bottom"))

                    # top
                    id = file_id_info.get_id('top')
                    top_file_name = "top_" + str(id) + '.jpg'
                    cv2.imwrite(top_file_name, roi_color_u)
                    file_id_info.set_id('top', id + 1)
                    # upload user's top cloth image into database and s3 storage
                    upload_s3.upload_user_image(top_file_name, "top", file_id_info.get_id("top"))
                    return result
                
            except ValueError as e:
                print("ERROR: " + str(e))

            cv2.imshow('camera-0', img)
    cap.release()
    cv2.destroyAllWindows()
