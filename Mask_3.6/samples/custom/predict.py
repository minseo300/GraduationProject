import os
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mrcnn.config import Config

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import cloth

#Training Model 검사를 위한 py
#%matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


class ClothConfig(Config):

    NAME = "cloth"

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 3  # Background + balloon

    STEPS_PER_EPOCH = 10

    DETECTION_MIN_CONFIDENCE = 0.9

config = cloth.ClothConfig()
Cloth_dir = "C:\\Users\\llod\\PycharmProjects\\Mask_3.6\\Mask_RCNN\\samples\\custom\\dataset"

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

# Load validation dataset
dataset = cloth.ClothDataset()
dataset.load_cloth(Cloth_dir, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}/nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

weights_path = "C:/Users/llod/PycharmProjects/Mask_3.6/Mask_RCNN/mask_rcnn_cloth_0005.h5"

model.load_weights(weights_path, by_name=True)

image_id = 4
#for i in range(5):
#image_id = i
image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       dataset.image_reference(image_id)))



# Run object detection
results = model.detect([image], verbose=1) # Object를 Detection 함


# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy. # 평가함수
image_ids = np.random.choice(dataset.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =  modellib.load_image_gt(dataset, InferenceConfig, image_id)

    molded_images = np.expand_dims(modellib.mold_image(image, InferenceConfig), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

    print("mAP: ", np.mean(APs))


    # Display results
    ax = get_ax(1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'], ax=ax,
                                title="Predictions")

    #print("masks", r['masks'])
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    display_images([image])

    splash = cloth.color_splash(image, r['masks'])
    display_images([splash], cols=1)
#
# mrcnn = model.run_graph([image], [
#     ("detections", model.keras_model.get_layer("mrcnn_detection").output),
#     ("masks", model.keras_model.get_layer("mrcnn_mask").output),
# ])
#
# # Get detection class IDs. Trim zero padding.
# det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
# # det_class_ids는 detection한 object수만큼 list가 생성되며 각 리스트에는 분류될 class 수(3)가 들어간다.
# det_count = np.where(det_class_ids == 0)[0][0]
# det_class_ids = det_class_ids[:det_count]
#
# det_boxes = utils.denorm_boxes(mrcnn["detections"][0, :, :4], image.shape[:2])
# det_mask_specific = np.array([mrcnn["masks"][0, i, :, :, c]
#                               for i, c in enumerate(det_class_ids)])
# det_masks = np.array([utils.unmold_mask(m, det_boxes[i], image.shape)
#                       for i, m in enumerate(det_mask_specific)])
# log("det_mask_specific", det_mask_specific)
# log("det_masks", det_masks)
#
# display_images(det_masks[:4] * 255, cmap="Blues_r", interpolation="none") # display_images -> model함수이며, detection한 Object 출력
# display_images(np.transpose(gt_mask, [2, 0, 1])) # detection한 Object 출력