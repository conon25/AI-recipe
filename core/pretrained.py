import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import re
import numpy as np
import os
import time
import json
import numpy
from glob import glob
from PIL import Image
import pickle
import matplotlib
matplotlib.use("Agg")


IMAGE_DIMS = (299, 299, 3)
	
def Augmentation(img):
	image = img
	x = image.reshape((1,) + image.shape)
	datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
	i = 0
	lst = []
	for batch in datagen.flow(x,batch_size=3):
		i += 1
		if i > 3: break  # 이미지 20장을 생성하고 마칩니다
		batch = batch.reshape(299,299,3)
		lst.append(batch)
	return lst

def Standardization(encode_train, standard):
    def image_load(image_path):
        # img = tf.io.read_file(image_path)
        # img = tf.image.decode_jpeg(img, channels=3)
        # img = tf.image.resize(img, (299, 299))
        print(image_path)
        imagePath ='dataset\\'+image_path
        print(imagePath)
        img = cv2.imread(imagePath)
        img = cv2.resize(img, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        img = img_to_array(img)
        if standard:
            img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    print(image_dataset)
    image_dataset = image_dataset.map(image_load)
    return image_dataset

# # Select the first 30000 captions from the shuffled set
image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


with open('test.json', 'r',encoding="utf-8") as f:
    json_data = json.load(f)
imagePaths = sorted(list(json_data.keys()))

path = 'dataset\\'
data = []
for imagePath in imagePaths:
	imagePath = path+imagePath
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)
	# extract set of class labels from the image path and update the
	# # labels list

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float32") / 255.0

datagen = ImageDataGenerator(rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

for img in tqdm(data):
    image = img
    x = tf.reshape(image,(1,299,299,3))
    image_features_extract_model(x)
    cnt = 0
    # for batch in datagen.flow(x,batch_size=1):
    #     if cnt==3:break
    #     image = tf.reshape(batch,(1,299,299,3))
    #     batch_features = image_features_extract_model(image)
    #     cnt+=1

# for img in tqdm(data):
#     for image in Augmentation(img):
#         image = tf.reshape(image,(1,299,299,3))
#         batch_features = image_features_extract_model(image)
#         batch_features = tf.reshape(batch_features,(-1, batch_features.shape[3]))

image_features_extract_model.save('image_features_extract_model.h5')