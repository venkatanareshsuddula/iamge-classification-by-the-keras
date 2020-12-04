import cv2

import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#CATEGORIES = ["Dog", "Cat"]
CATEGORIES = ["sundar", "elon"]



def prepare(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = img_array/255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)



model = tf.keras.models.load_model("64x3-CNN.model")

prediction = model.predict([prepare('/home/venkat/Desktop/ai/7.jpg')])
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0])])
