# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask
from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from pywebio.input import *
from pywebio.output import *
import argparse
from pywebio import start_server

"""
Do further Imports Here

"""

import tensorflow as tf
from keras.models import load_model

model = load_model('brain_hemorrh_vgg16_backbone_1.hdf5', compile = False)

app = Flask(__name__)

IMG_SIZE = 512

def predict():
    
    input_image_file = file_upload("Upload brain CT Scan", accept="image/*")
    image = tf.keras.preprocessing.image.img_to_array(input_image_file)
    
    image = image/255
    image = tf.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.expand_dims(image, axis=0)
    
    prediction = model.predict(image)
    output_mask = tf.squeeze(tf.squeeze(prediction, axis=0), axis=-1)
    output_mask = tf.image.grayscale_to_rgb(output_mask)
    output_image = tf.keras.preprocessing.image.array_to_img(output_mask)
    
    put_image(output_image)
    
app.add_url_rule('/tool', 'webio_view', webio_view(predict), methods=['GET', 'POST', 'OPTIONS'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()
    
    start_server(predict, port=args.port)
        