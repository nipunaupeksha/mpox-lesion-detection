from flask import Blueprint, render_template, flash, request, redirect, url_for 
from werkzeug.utils import secure_filename
from .utils import allowed_file, load_models, get_img_array, make_gradcam_heatmap, save_and_display_gradcam, process_cnn_models, save_original_file, get_image_path, get_ensemble_derma_information
import os
import tensorflow as tf
import pandas as pd
from .constants import GRADCAM_FOLDER, IMG_SIZE, LAST_CONV_LAYER_NAME, DERMA_DATA_PATH, CNN_MODELS, RNN_MODELS, GNN_MODELS

# Create a blueprint for the views
views = Blueprint('views', __name__)

# Load models
cnn_models = load_models(CNN_MODELS)
rnn_models = load_models(RNN_MODELS)
gnn_models = load_models(GNN_MODELS)

# Print models
print("CNN Models: ", cnn_models)
print("RNN Models: ", rnn_models)
print("GNN Models: ", gnn_models)

# Load derma data
df = pd.read_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)), DERMA_DATA_PATH))
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
        # Upload the image
        filename = secure_filename(file.filename)
        # Save the uploaded file
        save_original_file(file)
        # Get image path
        img_path = get_image_path(file)
        # Get CNN predictions
        cnn_predictions, cnn_ensembled_derma, cnn_values = process_cnn_models(cnn_models, img_path, file)
        # Get ensemble derma information for CNN
        cnn_derma_data = get_ensemble_derma_information(cnn_ensembled_derma, df_arr)

        # gradcam
        model = cnn_models["mobilenetv2"]
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        model.layers[-1].activation=None
        img_array = preprocess_input(get_img_array(img_path, size=IMG_SIZE))
        heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)
        cam_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),GRADCAM_FOLDER, secure_filename(file.filename))
        cam_path = save_and_display_gradcam(img_path, heatmap, cam_path)
        return render_template("index.html", filename=filename, predictions=cnn_predictions, values=cnn_values, predicted_derma=cnn_ensembled_derma, derma_data=cnn_derma_data)
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