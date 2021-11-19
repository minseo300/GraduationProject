import cv2
from PIL import Image
import numpy as np
from numpy import dot
from skimage.segmentation import clear_border
import skimage.morphology as mp
import scipy.ndimage.morphology as sm

# def make_mask(image):
#     th,img1=cv2.threshold(image,0,255,cv2.THRESH_OTSU)
#     img1=255-img1
#     img2=clear_border(img1)
#     disk2=mp.disk(2)
#     img2=mp.binary_dilation(img2,disk2)
#     img2=mp.binary_erosion(img2,disk2)
#     img2=mp.binary_opening(img2,disk2)
#     img2=mp.binary_closing(img1,disk2)
#
#     img2=sm.binary_fill_holes(img2)
#     img2=Image.fromarray(img2)
#     img2.save("mask_7.jpg")
#
#
#
# img=cv2.imread('styling_7_top.jpg',0) # original image
# make_mask(img) # create mask image
#
# # 배경 제거
# img_org=Image.open('styling_7_top.jpg')
# img_mask=Image.open('mask_7.jpg')
#
# img_mask=img_mask.convert('L') # grayscale
#
# img_org.putalpha(img_mask) # add alpha channel
# img_org.save('result_7.png') # result

###################################################################################
# # 이미지 불러오기
# img = cv2.imread('image4.jpg')
#
# # 변환 graky
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # 임계값 조절
# mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
#
# # mask
# mask = 255 - mask
#
# # morphology 적용
# # borderconstant 사용
# kernel = np.ones((3,3), np.uint8)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
# # anti-alias the mask
# # blur alpha channel
# mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
#
# # linear stretch so that 127.5 goes to 0, but 255 stays 255
# mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)
#
# # put mask into alpha channel
# result = img.copy()
# result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
# result[:, :, 3] = mask
#
# # 저장
# cv2.imwrite('translated.png', result)

cap = cv2.imread('image4.jpg')

# 배경 제거 객체 생성 --- ①
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
fgmask = fgbg.apply(cap)
# cv2.imshow('frame',cap)
# cv2.imshow('bgsub',fgmask)
cv2.imwrite('translated.png', fgmask)