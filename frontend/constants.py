# Allowed image extensions
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# Directory paths
MODELS_PATH = "./static/models"
UPLOAD_FOLDER="./static/uploads"
GRADCAM_FOLDER = './static/uploads/gradcam'
CNN_MODELS = MODELS_PATH + "/cnn"
RNN_MODELS = MODELS_PATH + "/rnn"
GNN_MODELS = MODELS_PATH + "/gnn"

# Derma dataset path
DERMA_DATA_PATH = './static/derma_dataset.csv'

# Last convolutional layer name for Grad-CAM
LAST_CONV_LAYER_NAME="Conv_1"

# Resizing dimensions for input images
IMG_SIZE=(224,224)

# Class labels for the model predictions
CLASSES = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox']