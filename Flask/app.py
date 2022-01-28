import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
global graph
#graph = tf.get_default_graph()
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import cv2

app = Flask(__name__)
model = load_model("cancer.h5")

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
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (128,128))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        

        preds = model.predict(x)
        preds = np.argmax(model.predict(x), axis=1)


        if preds==0:
            text = "Cancer is seen. We recommend you to get in touch with an oncologist at the earliest."
        else:
            text = "Cancer not seen. Stay safe and healthy."

        return text


if __name__ == '__main__':
    app.run(debug=False)
        
        
        
    
    
    