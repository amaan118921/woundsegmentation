from flask import Flask, request
from flask import jsonify

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

initialized = False

# This method does the initializations related to the firebase storage
def init():
    global initialized
    if not initialized:
        cred = credentials.Certificate('./womensafety-c4d41-1573ac3bb347.json')
        initialize_app(cred, {'storageBucket': 'womensafety-c4d41.appspot.com'})
        initialized = True

# The upload Image method uploads the resultant image to the cloud and returns the url of the image as the final
# response of the GET Request
def upload_img(filename, file):
    init()
    bucket = storage.bucket()
    blob = bucket.blob(file)
    blob.upload_from_filename(filename)

    blob.make_public()

    img_url = blob.public_url

    return jsonify({'resultUrl': img_url})

# This method first downloads the image from the given url and then feeds it into the cnn model.
# Then the model detects the contour and uploads the resultant image using the upload_img method.
def predict_and_save(image_url, filename):
    response = requests.get(image_url)
    if response.status_code == 200:
        image_data = response.content
        input_image = Image.open(BytesIO(image_data))
        input_image = input_image.resize((input_dim_x, input_dim_y))
        input_image = np.array(input_image) / 255.0  # Normalize the image (assuming pixel values are in [0, 255])
        model = load_model('./training_history/' + weight_file_name
                           , custom_objects={'recall': recall,
                                             'precision': precision,
                                             'dice_coef': dice_coef,
                                             'relu6': relu6,
                                             'DepthwiseConv2D': DepthwiseConv2D,
                                             'BilinearUpsampling': BilinearUpsampling})
        try:
            prediction = model.predict(np.expand_dims(input_image, axis=0))
            test_label_filenames_list = [filename]
            save_results(prediction, 'rgb', outputPath, test_label_filenames_list)
            res = upload_img(outputPath + filename, 'images/' + filename)
            return res
        except Exception as e:
            print("Error:", str(e))
            return jsonify({'resultUrl': None, 'error': str(e)})

    else:
        print("Failed to download the image from the URL: {image_url}")
        return jsonify({'resultUrl': None})


# endpoint for testing
@app.route('/')
def hello_world():
    return 'hello, world!'


# endpoint to detect the contour of the image, by extracting the url of the image and filename from the request.
@app.route('/predict', methods=['GET'])
def predict():
    url = request.args.get('url')
    filename = request.args.get('filename')
    # data = request.get_json()
    # url = data['url']
    # filename = data['filename']
    # url = 'https://firebasestorage.googleapis.com/v0/b/womensafety-c4d41.appspot.com/o/uploads%2Ffoot-ulcer-0027.png?alt=media&token=51790edf-d836-4c44-9c3d-e4c7eb72e5ad'
    # filename = "test.png"
    res = predict_and_save(url, filename)
    return res


if __name__ == '__main__':
    app.run(debug=True)
