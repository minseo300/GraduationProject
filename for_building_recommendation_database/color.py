import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from scipy.spatial import distance

black = [0,0,0]
silver = [192,192,192]
gray = [128,128,128]
white = [255,255,255]
maroon = [128,0,0]
red = [255,0,0]
purple = [128,0,128]
fuchsia = [255,0,255]
green = [0,128,0]
lime = [0,255,0]
olive = [128,128,0]
yellow = [255,255,0]
navy = [0,0,128]
blue = [0,0,255]
teal = [0,128,128]
aqua = [0,255,255]
aliceblue = [240,248,255]
antiquewhite = [250,235,215]
aquamarine = [127,255,212]
azure = [240,255,255]
beige = [245,245,220]
bisque = [255,228,196]
black = [0,0,0]
blanchedalmond = [255,235,205]
blue = [0,0,255]
blueviolet = [138,43,226]
brown = [165,42,42]
burlywood = [222,184,135]
cadetblue = [95,158,160]
chartreuse = [127,255,0]
chocolate = [210,105,30]
coral = [255,127,80]
cornflowerblue = [100,149,237]
cornsilk = [255,248,220]
crimson = [220,20,60]
cyan = [0,255,255]
darkblue = [0,0,139]
darkcyan = [0,139,139]
darkgoldenrod = [184,134,11]
darkgray = [169,169,169]
darkgreen = [0,100,0]
darkgrey = [169,169,169]
darkkhaki = [189,183,107]
darkmagenta = [139,0,139]
darkolivegreen = [85,107,47]
darkorange = [255,140,0]
darkorchid = [153,50,204]
darkred = [139,0,0]
darksalmon = [233,150,122]
darkseagreen = [143,188,143]
darkslateblue = [72,61,139]
darkslategray = [47,79,79]
darkslategrey = [47,79,79]
darkturquoise = [0,206,209]
darkviolet = [148,0,211]
deeppink = [255,20,147]
deepskyblue = [0,191,255]
dimgray = [105,105,105]
dimgrey = [105,105,105]
dodgerblue = [30,144,255]
firebrick = [178,34,34]
floralwhite = [255,250,240]
forestgreen = [34,139,34]
fuchsia = [255,0,255]
gainsboro = [220,220,220]
ghostwhite = [248,248,255]
gold = [255,215,0]
goldenrod = [218,165,32]
gray = [128,128,128]
green = [0,128,0]
greenyellow = [173,255,47]
grey = [128,128,128]
honeydew = [240,255,240]
hotpink = [255,105,180]
indianred = [205,92,92]
indigo = [75,0,130]
ivory = [255,255,240]
khaki = [240,230,140]
lavender = [230,230,250]
lavenderblush = [255,240,245]
lawngreen = [124,252,0]
lemonchiffon = [255,250,205]
lightblue = [173,216,230]
lightcoral = [240,128,128]
lightcyan = [224,255,255]
lightgoldenrodyellow = [250,250,210]
lightgray = [211,211,211]
lightgreen = [144,238,144]
lightgrey = [211,211,211]
lightpink = [255,182,193]
lightsalmon = [255,160,122]
lightseagreen = [32,178,170]
lightskyblue = [135,206,250]
lightslategray = [119,136,153]
lightslategrey = [119,136,153]
lightsteelblue = [176,196,222]
lightyellow = [255,255,224]
lime = [0,255,0]
limegreen = [50,205,50]
linen = [250,240,230]
magenta = [255,0,255]
maroon = [128,0,0]
mediumaquamarine = [102,205,170]
mediumblue = [0,0,205]
mediumorchid = [186,85,211]
mediumpurple = [147,112,219]
mediumseagreen = [60,179,113]
mediumslateblue = [123,104,238]
mediumspringgreen = [0,250,154]
mediumturquoise = [72,209,204]
mediumvioletred = [199,21,133]
midnightblue = [25,25,112]
mintcream = [245,255,250]
mistyrose = [255,228,225]
moccasin = [255,228,181]
navajowhite = [255,222,173]
navy = [0,0,128]
oldlace = [253,245,230]
olive = [128,128,0]
olivedrab = [107,142,35]
orange = [255,165,0]
orangered = [255,69,0]
orchid = [218,112,214]
palegoldenrod = [238,232,170]
palegreen = [152,251,152]
paleturquoise = [175,238,238]
palevioletred = [219,112,147]
papayawhip = [255,239,213]
peachpuff = [255,218,185]
peru = [205,133,63]
pink = [255,192,203]
plum = [221,160,221]
powderblue = [176,224,230]
purple = [128,0,128]
red = [255,0,0]
rosybrown = [188,143,143]
royalblue = [65,105,225]
saddlebrown = [139,69,19]
salmon = [250,128,114]
sandybrown = [244,164,96]
seagreen = [46,139,87]
seashell = [255,245,238]
sienna = [160,82,45]
silver = [192,192,192]
skyblue = [135,206,235]
slateblue = [106,90,205]
slategray = [112,128,144]
slategrey = [112,128,144]
snow = [255,250,250]
springgreen = [0,255,127]
steelblue = [70,130,180]
tan = [210,180,140]
teal = [0,128,128]
thistle = [216,191,216]
tomato = [255,99,71]
turquoise = [64,224,208]
violet = [238,130,238]
wheat = [245,222,179]
white = [255,255,255]
whitesmoke = [245,245,245]
yellow = [255,255,0]
yellowgreen = [154,205,50]


colordict = {
    'black' : black,
    'silver' : silver,
    'gray' : gray,
    'white' : white,
    'maroon' : maroon,
    'red' : red,
    'purple' : purple,
    'fuchsia' : fuchsia,
    'green' : green,
    'lime' : lime,
    'olive' : olive,
    'yellow' : yellow,
    'navy' : navy,
    'blue' : blue,
    'teal' : teal,
    'aqua' : aqua,
    'aliceblue' : aliceblue,
    'antiquewhite' : antiquewhite,
    'aquamarine' : aquamarine,
    'azure' : azure,
    'beige' : beige,
    'bisque' : bisque,
    'black' : black,
    'blanchedalmond' : blanchedalmond,
    'blue' : blue,
    'blueviolet' : blueviolet,
    'brown' : brown,
    'burlywood' : burlywood,
    'cadetblue' : cadetblue,
    'chartreuse' : chartreuse,
    'chocolate' : chocolate,
    'coral' : coral,
    'cornflowerblue' : cornflowerblue,
    'cornsilk' : cornsilk,
    'crimson' : crimson,
    'cyan' : cyan,
    'darkblue' : darkblue,
    'darkcyan' : darkcyan,
    'darkgoldenrod' : darkgoldenrod,
    'darkgray' : darkgray,
    'darkgreen' : darkgreen,
    'darkgrey' : darkgrey,
    'darkkhaki' : darkkhaki,
    'darkmagenta' : darkmagenta,
    'darkolivegreen' : darkolivegreen,
    'darkorange' : darkorange,
    'darkorchid' : darkorchid,
    'darkred' : darkred,
    'darksalmon' : darksalmon,
    'darkseagreen' : darkseagreen,
    'darkslateblue' : darkslateblue,
    'darkslategray' : darkslategray,
    'darkslategrey' : darkslategrey,
    'darkturquoise' : darkturquoise,
    'darkviolet' : darkviolet,
    'deeppink' : deeppink,
    'deepskyblue' : deepskyblue,
    'dimgray' : dimgray,
    'dimgrey' : dimgrey,
    'dodgerblue' : dodgerblue,
    'firebrick' : firebrick,
    'floralwhite' : floralwhite,
    'forestgreen' : forestgreen,
    'fuchsia' : fuchsia,
    'gainsboro' : gainsboro,
    'ghostwhite' : ghostwhite,
    'gold' : gold,
    'goldenrod' : goldenrod,
    'gray' : gray,
    'green' : green,
    'greenyellow' : greenyellow,
    'grey' : grey,
    'honeydew' : honeydew,
    'hotpink' : hotpink,
    'indianred' : indianred,
    'indigo' : indigo,
    'ivory' : ivory,
    'khaki' : khaki,
    'lavender' : lavender,
    'lavenderblush' : lavenderblush,
    'lawngreen' : lawngreen,
    'lemonchiffon' : lemonchiffon,
    'lightblue' : lightblue,
    'lightcoral' : lightcoral,
    'lightcyan' : lightcyan,
    'lightgoldenrodyellow' : lightgoldenrodyellow,
    'lightgray' : lightgray,
    'lightgreen' : lightgreen,
    'lightgrey' : lightgrey,
    'lightpink' : lightpink,
    'lightsalmon' : lightsalmon,
    'lightseagreen' : lightseagreen,
    'lightskyblue' : lightskyblue,
    'lightslategray' : lightslategray,
    'lightslategrey' : lightslategrey,
    'lightsteelblue' : lightsteelblue,
    'lightyellow' : lightyellow,
    'lime' : lime,
    'limegreen' : limegreen,
    'linen' : linen,
    'magenta' : magenta,
    'maroon' : maroon,
    'mediumaquamarine' : mediumaquamarine,
    'mediumblue' : mediumblue,
    'mediumorchid' : mediumorchid,
    'mediumpurple' : mediumpurple,
    'mediumseagreen' : mediumseagreen,
    'mediumslateblue' : mediumslateblue,
    'mediumspringgreen' : mediumspringgreen,
    'mediumturquoise' : mediumturquoise,
    'mediumvioletred' : mediumvioletred,
    'midnightblue' : midnightblue,
    'mintcream' : mintcream,
    'mistyrose' : mistyrose,
    'moccasin' : moccasin,
    'navajowhite' : navajowhite,
    'navy' : navy,
    'oldlace' : oldlace,
    'olive' : olive,
    'olivedrab' : olivedrab,
    'orange' : orange,
    'orangered' : orangered,
    'orchid' : orchid,
    'palegoldenrod' : palegoldenrod,
    'palegreen' : palegreen,
    'paleturquoise' : paleturquoise,
    'palevioletred' : palevioletred,
    'papayawhip' : papayawhip,
    'peachpuff' : peachpuff,
    'peru' : peru,
    'pink' : pink,
    'plum' : plum,
    'powderblue' : powderblue,
    'purple' : purple,
    'red' : red,
    'rosybrown' : rosybrown,
    'royalblue' : royalblue,
    'saddlebrown' : saddlebrown,
    'salmon' : salmon,
    'sandybrown' : sandybrown,
    'seagreen' : seagreen,
    'seashell' : seashell,
    'sienna' : sienna,
    'silver' : silver,
    'skyblue' : skyblue,
    'slateblue' : slateblue,
    'slategray' : slategray,
    'slategrey' : slategrey,
    'snow' : snow,
    'springgreen' : springgreen,
    'steelblue' : steelblue,
    'tan' : tan,
    'teal' : teal,
    'thistle' : thistle,
    'tomato' : tomato,
    'turquoise' : turquoise,
    'violet' : violet,
    'wheat' : wheat,
    'white' : white,
    'whitesmoke' : whitesmoke,
    'yellow' : yellow,
    'yellowgreen' : yellowgreen
}

def find_color(color):
    min = 100000
    mincolor="default"
    for key, value in colordict.items():
        dst = distance.euclidean(color, value)
        if min>dst:
            min = dst
            mincolor = key
    return mincolor

def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar
image=Image.open('E:/casual/7/styling_2_outer.jpg').convert('RGB')
image.show()
image = cv2.imread("E:/casual/7/styling_2_outer.jpg", cv2.IMREAD_UNCHANGED)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = image.reshape((image.shape[0] * image.shape[1], 3))  # height, width ??????

k = 3
clt = KMeans(n_clusters=k)
clt.fit(image)
hist = centroid_histogram(clt)
back = 0
for i in range(3):
    print(i, hist[i], find_color(clt.cluster_centers_[i]))
    if hist[i] < 0.1:
        continue
    if find_color(clt.cluster_centers_[i]) == "??????":
        if back == 1:
            print("??????")
            break
        back = 1
    else:
        print('final: ',find_color(clt.cluster_centers_[i]))
        break

    # bar ????????????

bar = plot_colors(hist, clt.cluster_centers_)

# #????????? ????????? ????????? ????????? ????????? ??? ?????????
# image_dir = "C:/Users/llod/Desktop/m/street"
# image_list = os.listdir(image_dir)
#
#
#
#
# for image_name in image_list:
#     image_path = image_dir+'/'+ image_name
#     print(image_path)
#     image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     image = image.reshape((image.shape[0] * image.shape[1], 3)) # height, width ??????
#
#     k = 3
#     clt = KMeans(n_clusters = k)
#     clt.fit(image)
#     hist = centroid_histogram(clt)
#     back = 0
#     for i in range(3):
#         print(i, hist[i], find_color(clt.cluster_centers_[i]))
#         if hist[i] <0.1:
#             continue
#         if find_color(clt.cluster_centers_[i]) == "??????":
#             if back == 1:
#                 print("??????")
#                 break
#             back = 1
#         else :
#             print(find_color(clt.cluster_centers_[i]))
#             break
#
#
#     #bar ????????????
#
#     bar = plot_colors(hist, clt.cluster_centers_)

    # # show our color bart
    # plt.figure()
    # plt.axis("off")
    # plt.imshow(bar)
    # plt.show()