import os
import tensorflow as tf
import numpy as np
import matplotlib.cm as cm

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
MODELS_PATH = "./static/models"

# check whether the uploaded image has the correct format
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Load all the models from the models directory
def load_models():
    models = dict()

    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), MODELS_PATH)
    files = os.listdir(path)

    # Add models to the list
    for index, file in enumerate(files):
        if '.DS_Store' in file:
            continue
        name = file.rsplit(".", 1)[0].lower().split("_")[-1]
        # Load tf and torch models
        # model = 
        models[name] = model
    return models

def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array to a "batch" of size "size"
    array = np.expand_dims(array, axis=0)
    return array