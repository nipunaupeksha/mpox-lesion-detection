from flask import Blueprint, render_template, flash, request, redirect, url_for 
from werkzeug.utils import secure_filename
from finfacts.utils import allowed_file, load_models, get_img_array, make_gradcam_heatmap, save_and_display_gradcam, most_common
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

views = Blueprint('views', __name__)

# load models
models = load_models()

# load derma data
df = pd.read_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)), './static/derma_dataset.csv'))
df_arr = df.values.tolist()

UPLOAD_FOLDER="./static/uploads"
GRADCAM_FOLDER = './static/uploads/gradcam'
LAST_CONV_LAYER_NAME="Conv_1"
IMG_SIZE=(224,224)
CLASSES = ['Monkeypox']

# main route to show the upload image and get image classification
@views.route('/')
def home():
    return render_template("index.html")

@views.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash("No file part")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        # upload
        filename = secure_filename(file.filename)
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),UPLOAD_FOLDER,secure_filename(file.filename)))
        # get all predictions
        img_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),UPLOAD_FOLDER, secure_filename(file.filename))
        img = plt.imread(img_path)
        img = cv2.resize(img, (224,224))
        img = np.expand_dims(img, axis=0)
        predictions = dict()
        values = dict() # these are model values
        derma_data = [] # get derma data
        for key in models:
            model = models[key]
            arr = model.predict(img)
            print(key,arr)
            index = np.argmax(arr, axis=1)[0]
            print(index)
            values[key] = round((arr[0][index] * 100),2)
            predictions[key] = CLASSES[index]
        predicted_derma = most_common(list(predictions.values()))
        index = 0
        for i in range(len(CLASSES)):
            if predicted_derma in CLASSES[i]:
                index = i
                break
        derma_data = df_arr[index]
        # TODO not locations something else
        # get location based derma
        derma_locations = derma_data[3].split(",")
        location_derma = dict()
        for location in derma_locations:
            location_derma[location] = []
        for i in range(len(df_arr)):
            if predicted_derma in df_arr[i][0]:
                continue
            for location in derma_locations:
                if location.strip() in df_arr[i][3]: # location based derma
                    temp_lst = []
                    temp_lst.append(df_arr[i][0])
                    temp_lst.append(os.path.join(f"derma/{i+1}.jpg"))
                    location_derma[location].append(temp_lst)
        # print(location_derma)
        # gradcam
        model = models["mobilenetv2"]
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        model.layers[-1].activation=None
        img_array = preprocess_input(get_img_array(img_path, size=IMG_SIZE))
        heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)
        cam_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),GRADCAM_FOLDER, secure_filename(file.filename))
        cam_path = save_and_display_gradcam(img_path, heatmap, cam_path)
        return render_template("index.html", filename=filename, predictions=predictions, values=values, predicted_derma=predicted_derma, derma_data=derma_data, location_derma=location_derma, derma_locations=derma_locations)
    else:
        flash("Allowed image types are .png, .jpg and .jpeg")
        return redirect(request.url)

@views.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@views.route('/display/gradcam/<filename>')
def display_gradcam(filename):
    return redirect(url_for('static', filename='uploads/gradcam/'+filename), code=301)