# -*- coding: utf-8 -*-
"""
Created on Sat May 30 17:01:47 2020

@author: LINGAM
"""


import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
global graph
graph = tf.compat.v1.get_default_graph()
from flask import Flask , request,  render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.python.keras.backend import set_session
app = Flask(__name__)

sess=tf.compat.v1.Session(   )
set_session(sess)
model = load_model("qurat.h5")
@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',secure_filename(f.filename))
        print("upload folder is ", filepath)
        f.save(filepath)
        
        
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis = 0)
        
        
        #with session.as_default():
        with graph.as_default():
            preds = model.predict_classes(x)
            print(preds)
            index =["0","1","2","3","4","5","6","7","8","9"]
        
        text = "The hand gesture is - " +index[preds[0]]
        
    return text


if __name__ == '__main__':
    app.run(debug = True, port=5000, host="localhost")