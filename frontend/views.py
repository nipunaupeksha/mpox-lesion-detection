from flask import Blueprint, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from .utils import allowed_file, load_models, get_img_array, make_gradcam_heatmap, save_and_display_gradcam, most_common
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

views = Blueprint("views", __name__)

# Load the models
models = load_models()