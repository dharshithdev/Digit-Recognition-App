from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('../model/mnist_model.h5')


def preprocess_image(file):
    # Read image from request
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Resize
    img = cv2.resize(img, (28, 28))

    # Invert colors
    img = 255 - img

    # Blur (VERY IMPORTANT)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Normalize
    img = img / 255.0

    img = img.reshape(1, 28, 28)

    return img


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']

        processed_img = preprocess_image(file)

        prediction = model.predict(processed_img)
        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return jsonify({
            'digit': digit,
            'confidence': round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)