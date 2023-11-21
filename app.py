from flask import Flask, request

from io import BytesIO
from firebase_admin import credentials, initialize_app, storage
import requests
from PIL import Image
from keras.models import load_model

from models.deeplab import relu6, BilinearUpsampling, DepthwiseConv2D

from utils.learning.metrics import dice_coef, precision, recall
from utils.io.data import save_results
import numpy as np

app = Flask(__name__)

input_dim_x = 224
input_dim_y = 224
color_space = 'rgb'
path = './data/Medetec_foot_ulcer_224/'
outputPath = './data/output/'
weight_file_name = 'test.hdf5'
pred_save_path = 'test/'

model = load_model('./training_history/' + weight_file_name
                   , custom_objects={'recall': recall,
                                     'precision': precision,
                                     'dice_coef': dice_coef,
                                     'relu6': relu6,
                                     'DepthwiseConv2D': DepthwiseConv2D,
                                     'BilinearUpsampling': BilinearUpsampling})

initialized = False


def init():
    global initialized
    if not initialized:
        cred = credentials.Certificate('./womensafety-c4d41-1573ac3bb347.json')
        initialize_app(cred, {'storageBucket': 'womensafety-c4d41.appspot.com'})
        initialized = True


def upload_img(filename, file):
    init()
    bucket = storage.bucket()
    blob = bucket.blob(file)
    blob.upload_from_filename(filename)

    blob.make_public()

    print("your file url", blob.public_url)


def predict_result(image_url, filename):
    response = requests.get(image_url)
    if response.status_code == 200:
        image_data = response.content
        input_image = Image.open(BytesIO(image_data))
        input_image = input_image.resize((input_dim_x, input_dim_y))
        input_image = np.array(input_image) / 255.0  # Normalize the image (assuming pixel values are in [0, 255])

        # Predict using the model
        prediction = model.predict(np.expand_dims(input_image, axis=0))
        test_label_filenames_list = [filename]

        # Save the prediction result
        # save_results(prediction, 'rgb', outputPath, test_label_filenames_list)
        try:
            upload_img(outputPath + filename, 'images/' + filename)
            return True
        except Exception as e:
            print("Error:", str(e))
            return False

    else:
        print("Failed to download the image from the URL: {image_url}")
        return False


@app.route('/')
def hello_world():
    return 'hello, world!'


@app.route('/predict')
def predict():
    # url = request.args.get('url')
    # filename = request.args.get('filename')
    url = 'https://firebasestorage.googleapis.com/v0/b/womensafety-c4d41.appspot.com/o/uploads%2Ffoot-ulcer-0027.png?alt=media&token=51790edf-d836-4c44-9c3d-e4c7eb72e5ad'
    name = "foot-ulcer-0027.png"
    if predict_result(url, name): return 'success'
    return 'failed'
