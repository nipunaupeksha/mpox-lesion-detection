import os
import tensorflow as tf
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.cm as cm
from .constants import CLASSES

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# Directory paths
MODELS_PATH = "./static/models"
UPLOAD_FOLDER="./static/uploads"
GRADCAM_FOLDER = './static/uploads/gradcam'
CNN_MODELS = MODELS_PATH + "/cnn"
RNN_MODELS = MODELS_PATH + "/rnn"
GNN_MODELS = MODELS_PATH + "/gnn"

# Check whether the uploaded image has the correct format
def allowed_file(filename):
    # Check if the file has an allowed extension
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Save the uploaded file
def save_original_file(file):
    # Save the uploaded file in the UPLOAD_FOLDER and return the file path
    file_path = file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),UPLOAD_FOLDER, secure_filename(file.filename)))
    return file_path

# Get image path
def get_image_path(file):
    # Get the image path
    img_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),UPLOAD_FOLDER, secure_filename(file.filename))
    return img_path

# Load models
def load_models(model_path):
    models = dict()
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_path)
    files = os.listdir(path)

    # add models to the list
    for file in files:
        if '.DS_Store' in file:
            continue
        name = file.rsplit('.',1)[0].lower().split('_')[-1]
        # Load models
        if("cnn" in model_path or "rnn" in model_path or "gnn" in model_path):
            model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), model_path, file))
            models[name] = model
    return models

def load_preprocess_input_methods():
    preprocess_input = dict()
    preprocess_input["mobilenetv2"] = tf.keras.applications.mobilenet_v2.preprocess_input
    preprocess_input["inceptionresnetv2"] = tf.keras.applications.inception_resnet_v2.preprocess_input
    preprocess_input["inceptionv3"] = tf.keras.applications.inception_v3.preprocess_input
    preprocess_input["resnetv2"] = tf.keras.applications.resnet_v2.preprocess_input
    preprocess_input["vgg16"] = tf.keras.applications.vgg16.preprocess_input
    preprocess_input["vgg19"] = tf.keras.applications.vgg19.preprocess_input
    preprocess_input["xception"] = tf.keras.applications.xception.preprocess_input
    preprocess_input["gcn"] = tf.keras.applications.inception_resnet_v2.preprocess_input
    preprocess_input["dualgcn"] = tf.keras.applications.inception_resnet_v2.preprocess_input
    preprocess_input["gin"] = tf.keras.applications.inception_resnet_v2.preprocess_input
    preprocess_input["gat"] = tf.keras.applications.inception_resnet_v2.preprocess_input
    return preprocess_input

# Process CNN models
def process_cnn_models(cnn_models, img_path):
    # Load preprocess input
    preprocess_input=load_preprocess_input_methods()

    # Read and preprocess the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Save the predictions in a dictionary
    cnn_predictions = dict()
    cnn_values = dict()

    # Iterate through the models and make predictions
    for key in cnn_models:
        img_array_cnn = preprocess_input[key](img_array)
        model = cnn_models[key]
        arr = model.predict(img_array_cnn)
        index = np.argmax(arr[0])
        cnn_values[key] = round((arr[0][index] * 100),2)
        cnn_predictions[key] = CLASSES[index]

    # Ensemble the predictions
    cnn_ensembled_derma = ensembling_results(list(cnn_predictions.values()))

    # Return the predictions and ensemble result
    return cnn_predictions, cnn_ensembled_derma, cnn_values

# Process RNN models
def process_rnn_models(rnn_models, img_path):
    # Read and preprocess the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    img = img / 255.0
    reshaped_img = img.reshape((224, 224 * 3))
    img_array = np.expand_dims(reshaped_img, axis=0)

    # img_array = img_to_array(img)
    # img_array = np.expand_dims(img_array, axis=0)

    # Save the predictions in a dictionary
    rnn_predictions = dict()
    rnn_values = dict()

    # Iterate through the models and make predictions
    for key in rnn_models:
        model = rnn_models[key]
        arr = model.predict(img_array)
        index = np.argmax(arr[0])
        rnn_values[key] = round((arr[0][index] * 100), 2)
        rnn_predictions[key] = CLASSES[index]

    # Ensemble the predictions
    rnn_ensembled_derma = ensembling_results(list(rnn_predictions.values()))

    # Return the predictions and ensemble result
    return rnn_predictions, rnn_ensembled_derma, rnn_values

# Process GNN models
def process_gnn_models(gnn_models, img_path):
    # Load preprocess input
    preprocess_input=load_preprocess_input_methods()

    # Read and preprocess the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Save the predictions in a dictionary
    gnn_predictions = dict()
    gnn_values = dict()

    # Iterate through the models and make predictions
    for key in gnn_models:
        img_array_gnn = preprocess_input[key](img_array)
        model = gnn_models[key]
        arr = model.predict(img_array_gnn)
        index = np.argmax(arr[0])
        gnn_values[key] = round((arr[0][index] * 100),2)
        gnn_predictions[key] = CLASSES[index]

    # Ensemble the predictions
    gnn_ensembled_derma = ensembling_results(list(gnn_predictions.values()))

    # Return the predictions and ensemble result
    return gnn_predictions, gnn_ensembled_derma, gnn_values

# Get ensemble derma information
def get_ensemble_derma_information(cnn_ensembled_derma, df_arr):
    index = 0
    for i in range(len(CLASSES)):
        if cnn_ensembled_derma in CLASSES[i]:
            index = i
            break
    cnn_derma_data = df_arr[index]
    return cnn_derma_data

# Function to get the image array
def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array to a "batch" of size "size"
    array = np.expand_dims(array, axis=0)
    return array

# Function to create Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    # Then, we compute the gradient of the top predicted class for out input image with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen) with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array by "how important this channel is" with regard to the top predicted class
    # After that, the sum of all the channels are obtained for heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to save and display the Grad-CAM heatmap
def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    return cam_path

# Function to ensemble the results from different models
def ensembling_results(prediction_list):
    # Return the ensemble result with neural network type
    return max(set(prediction_list), key=prediction_list.count)
