import os
import shutil
import json
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def mergetSort(x):
    if len(x) <= 1:
        return x
    left = mergetSort(x[:len(x)//2])
    right = mergetSort(x[len(x)//2: ])

    i,j,k = 0,0,0
    while i<len(left) and j<len(right):
        if left[i] < right[j]:
            x[k] = left[i]
            i += 1
        else:
            x[k] = right[j]
            j += 1
            k += 1
    if i == len(left):
        while j < len(right):
            x[k] = right[j]
            j += 1
            k += 1
    elif j == len(right):
        while i < len(left):
            x[k] = left[i]
            i += 1
            k += 1
    return x

label_dir = "C:/Users/llod/Desktop/K-Fashion 이미지/Validation/valid_라벨링데이터/라벨링데이터"
label = os.listdir(label_dir)
image_dir = "C:/Users/llod/Desktop/K-Fashion 이미지/Validation/valid_원천데이터/원천데이터"
image = os.listdir(image_dir)
save_dir = "C:/Users/llod/Desktop/mrcnn"
VIA_dict = {}
isave_dir = "C:/Users/llod/Desktop/validate"
for file in label:
    style = os.listdir(label_dir+'/'+file)
    if file == "히피":
        for file_name in style:
            with open(label_dir + '/' +file+'/' + file_name, encoding='utf8', errors='ignore') as f:
                json_data = json.load(f)

                txlst = []
                tylst = []
                bxlst = []
                bylst = []
                oxlst = []
                oylst = []
                if json_data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['아우터'][0]:
                    continue

                image_name = json_data['이미지 정보']['이미지 파일명']
                image = Image.open(image_dir + '/' + file + '/' + image_name)
                image.save(isave_dir + '/' + image_name)
                image_size = image.size
                img_size = image_size[0] * image_size[1]

                VIA_dict[image_name] = {
                    "fileref": "",
                    "size": img_size,
                    "filename": image_name,
                    "base64_img_data": "",
                    "file_attributes": {},
                    "regions": {

                    }
                }

                if json_data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['상의'][0]:
                    top_polygon=json_data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['상의'][0]
                    top_sorted = sorted(top_polygon.items())
                    top_sorted_len = int(len(top_sorted)/2)
                    top_sorted = list_chunk(top_sorted,top_sorted_len)
                    top_sorted_x = top_sorted[0]
                    top_sorted_y = top_sorted[1]
                    for i in range(len(top_sorted_x)):
                        top_sorted_x[i] = list(top_sorted_x[i])
                        top_sorted_x[i][0] = top_sorted_x[i][0].replace('X좌표','')
                        top_sorted_x[i][0] = int(top_sorted_x[i][0])
                    top_sorted_x = sorted(top_sorted_x)
                    # print(top_sorted_x)
                    if len(top_sorted_y)!=len(top_sorted_x):
                        assert("slice error")

                    for i in range(len(top_sorted_y)):
                        top_sorted_y[i] = list(top_sorted_y[i])
                        top_sorted_y[i][0] = top_sorted_y[i][0].replace('Y좌표','')
                        top_sorted_y[i][0] = int(top_sorted_y[i][0])
                    top_sorted_y = sorted(top_sorted_y)
                    # print(top_sorted_y)

                    for i in top_sorted_x:
                        txlst.append(i[1])
                    for i in top_sorted_y:
                        tylst.append(i[1])


                    VIA_dict[image_name]['regions']['0'] = {
                            "shape_attributes": {
                                "name": "polygon",
                                "all_points_x": txlst,
                                "all_points_y": tylst},
                            "region_attributes": {"name":"top"}
                        }

                if json_data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['하의'][0]:
                    bot_polygon = json_data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['하의'][0]
                    bot_sorted = sorted(bot_polygon.items())
                    bot_sorted_len = int(len(bot_sorted) / 2)
                    bot_sorted = list_chunk(bot_sorted, bot_sorted_len)
                    bot_sorted_x = bot_sorted[0]
                    bot_sorted_y = bot_sorted[1]
                    for i in range(len(bot_sorted_x)):
                        bot_sorted_x[i] = list(bot_sorted_x[i])
                        bot_sorted_x[i][0] = bot_sorted_x[i][0].replace('X좌표', '')
                        bot_sorted_x[i][0] = int(bot_sorted_x[i][0])
                    bot_sorted_x = sorted(bot_sorted_x)
                    # print(bot_sorted_x)
                    if len(bot_sorted_y) != len(bot_sorted_x):
                        assert ("slice error")

                    for i in range(len(bot_sorted_y)):
                        bot_sorted_y[i] = list(bot_sorted_y[i])
                        bot_sorted_y[i][0] = bot_sorted_y[i][0].replace('Y좌표', '')
                        bot_sorted_y[i][0] = int(bot_sorted_y[i][0])
                    bot_sorted_y = sorted(bot_sorted_y)
                    # print(bot_sorted_y)

                    for i in bot_sorted_x:
                        bxlst.append(i[1])
                    for i in bot_sorted_y:
                        bylst.append(i[1])

                    VIA_dict[image_name]['regions']['1'] = {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": bxlst,
                            "all_points_y": bylst},
                        "region_attributes": {"name": "bottom"}
                    }

                if json_data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['원피스'][0]:
                    one_polygon = json_data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['원피스'][0]
                    one_sorted = sorted(one_polygon.items())
                    one_sorted_len = int(len(one_sorted) / 2)
                    one_sorted = list_chunk(one_sorted, one_sorted_len)
                    one_sorted_x = one_sorted[0]
                    one_sorted_y = one_sorted[1]
                    for i in range(len(one_sorted_x)):
                        one_sorted_x[i] = list(one_sorted_x[i])
                        one_sorted_x[i][0] = one_sorted_x[i][0].replace('X좌표', '')
                        one_sorted_x[i][0] = int(one_sorted_x[i][0])
                    one_sorted_x = sorted(one_sorted_x)
                    # print(one_sorted_x)
                    if len(one_sorted_y) != len(one_sorted_x):
                        assert ("slice error")

                    for i in range(len(one_sorted_y)):
                        one_sorted_y[i] = list(one_sorted_y[i])
                        one_sorted_y[i][0] = one_sorted_y[i][0].replace('Y좌표', '')
                        one_sorted_y[i][0] = int(one_sorted_y[i][0])
                    one_sorted_y = sorted(one_sorted_y)
                    # print(one_sorted_y)

                    for i in one_sorted_x:
                        oxlst.append(i[1])
                    for i in one_sorted_y:
                        oylst.append(i[1])

                    VIA_dict[image_name]['regions']['2'] = {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": oxlst,
                            "all_points_y": oylst},
                        "region_attributes": {"name": "onepiece"}
                    }


# print(type(VIA_dict))
with open(save_dir + '/via_region_data.json', 'w') as ff:
    json.dump(VIA_dict, ff, indent=4)
print("end")