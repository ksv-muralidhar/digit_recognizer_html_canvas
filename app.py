import numpy as np
import base64
import cv2
import os
from flask import Flask, render_template, request
from flask_cors import cross_origin, CORS
from tensorflow import keras

app = Flask(__name__)
CORS(app)
model = keras.models.load_model(os.path.join('models', 'mnist.h5'))
img = None


@app.route("/")
@cross_origin()
def index():
    return render_template("index.html")


@app.route("/", methods=["POST"])
@cross_origin()
def predict():
    global img
    try:
        img_b64 = request.form.get('hid')
        img_b64 = img_b64.replace('data:image/png;base64,', "")
        img_b64 = base64.b64decode(img_b64)
        nd = np.frombuffer(img_b64, np.uint8)
        img = cv2.imdecode(nd, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = img.reshape((1, 28, 28, 1))
        img = img / 255
        if img is not None:
            prediction = str(np.argmax(model.predict(img), axis=-1)[0])
            img = None
        else:
            prediction = -1
    except:
        prediction = -1
    return render_template('index.html', result=prediction)


if __name__ == "__main__":
    app.run()
