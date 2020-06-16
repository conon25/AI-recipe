import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io
import cv2
from keras.preprocessing.image import img_to_array

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
sys.path.append(os.path.join(ROOT_DIR, "samples\\food\\"))

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn import visualize
from mrcnn import model as modellib
from mrcnn.model import log

from samples.food import food

# matplotlib.use('tkagg')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

config = food.FoodConfig()
weights_path = "./mask_rcnn_food_0060.h5"

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

class_names = ['BG', '감자', '양파', '스팸', '계란', '고추', '사과']

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def distinct(image_b):
    image = cv2.imdecode(np.frombuffer(image_b, np.uint8), -1)
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    image = img_to_array(image)

    DEVICE = "/cpu:0"
    TEST_MODE = "inference"

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(weights_path, by_name=True)
    results = model.detect([image], verbose=1)

    # Display results
    ax = get_ax(1)
    r = results[0]

    masked_image = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    res = []
    s = r['scores'].tolist()
    j = 0
    for i in r['class_ids'].tolist() :
        res.append([class_names[i], s[j]])
        j+=1

    return masked_image, res





