# -*- coding: utf-8 -*-
"""
Author :- @Niraj.Bansal

"""

####################################################

""" Importing the Necessary libraries: - """ 
from flask import Flask
from pywebio.platform.flask import webio_view
from pywebio.input import *
from pywebio.output import *
import argparse
from pywebio import start_server

import tensorflow as tf
from tensorflow.keras.models import load_model

####################################################

""" Lets load in the trained attention_unet model, and define a flask app """

model = load_model('brain_hemorrh_vgg16_backbone_2.hdf5', compile = False)

app = Flask(__name__)

IMG_SIZE = 512   # Image is resized to input shape 512 X 512 X 3
THRESHOLD = 0.7  # Threshold to convert prediction in boolean format

###################################################

""" Defining the function to form predictions """

def predict():
    
    input_image_file = file_upload("Upload brain CT Scan", accept="image/*")
    image = tf.io.decode_image(input_image_file['content'], channels=3)   # It reads the bytes image, into tensor
    
    # Normalization, and preprocessing
    image = image/255 
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.expand_dims(image, axis=0)
    
    # Prediction, which is stored as bool tensor, based on threshold 
    prediction = model.predict(image)
    output_mask = tf.squeeze(prediction, axis=0)
    output_mask = tf.cast(output_mask > THRESHOLD, tf.float32)

    # If number of pixels predicted is less then 0.5% of total image size, no hemorrhage is detected
    
    if tf.reduce_sum(output_mask) > 0.005 * IMG_SIZE * IMG_SIZE:
        put_text("Major Hemorrhage Detected")
        output_mask = tf.image.grayscale_to_rgb(output_mask)
        output_image = tf.keras.preprocessing.image.array_to_img(output_mask)
        put_image(output_image)
        
    else:
        put_text("No major Hemorrhage Detected")
        
    
app.add_url_rule('/', 'webio_view', webio_view(predict), methods=['GET', 'POST', 'OPTIONS'])
        
app.run(host='localhost', port=80)
