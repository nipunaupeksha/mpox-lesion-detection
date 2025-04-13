from flask import Blueprint, render_template, flash, request, redirect, url_for 
from werkzeug.utils import secure_filename
from .utils import allowed_file, load_models, get_img_array, make_gradcam_heatmap, save_and_display_gradcam, ensembling_results,load_preprocess_input_methods, save_original_file, CNN_MODELS, RNN_MODELS, GNN_MODELS
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import pandas as pd

UPLOAD_FOLDER="./static/uploads"
GRADCAM_FOLDER = './static/uploads/gradcam'

LAST_CONV_LAYER_NAME="Conv_1"

IMG_SIZE=(224,224)
CLASSES = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox']

views = Blueprint('views', __name__)

# load models
cnn_models = load_models(CNN_MODELS)
rnn_models = load_models(RNN_MODELS)
gnn_models = load_models(GNN_MODELS)

# load preprocess input
cnn_preprocess_input = load_preprocess_input_methods('cnn')

# print models
print("CNN Models: ", cnn_models)

# load derma data
df = pd.read_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)), './static/derma_dataset.csv'))
df_arr = df.values.tolist()


# main route to show the upload image and get image classification
@views.route('/')
def home():
    # Render the home page
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

        # save the uploaded file
        save_original_file(file)
        # get CNN predictions
        img_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),UPLOAD_FOLDER, secure_filename(file.filename))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        cnn_predictions = dict()
        cnn_values = dict() # these are model values
        cnn_derma_data = [] # get derma data
        for key in cnn_models:
            img_array_cnn = cnn_preprocess_input[key](img_array)
            model = cnn_models[key]
            arr = model.predict(img_array_cnn)
            index = np.argmax(arr[0])
            cnn_values[key] = round((arr[0][index] * 100),2)
            cnn_predictions[key] = CLASSES[index]
        cnn_predicted_derma = ensembling_results(list(cnn_predictions.values()))
        index = 0
        for i in range(len(CLASSES)):
            if cnn_predicted_derma in CLASSES[i]:
                index = i
                break
        cnn_derma_data = df_arr[index]

        # gradcam
        model = cnn_models["mobilenetv2"]
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        model.layers[-1].activation=None
        img_array = preprocess_input(get_img_array(img_path, size=IMG_SIZE))
        heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)
        cam_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),GRADCAM_FOLDER, secure_filename(file.filename))
        cam_path = save_and_display_gradcam(img_path, heatmap, cam_path)
        return render_template("index.html", filename=filename, predictions=cnn_predictions, values=cnn_values, predicted_derma=cnn_predicted_derma, derma_data=cnn_derma_data)
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