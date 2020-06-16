# USAGE

# import the necessary packages
import keras
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from IPython import embed
# construct the argument parse and parse the arguments
def detect(image, model, mlb):
    embed()
    image = cv2.resize(image, (299, 299))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    embed()
    # model = load_model('fashion.model')
    # mlb = pickle.loads(open('mlb.pickle', "rb").read())

    proba = model.predict(image)[0]
    embed()
    return zip(mlb.classes_, proba)