from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import base64
import re

app = Flask(__name__)
model = load_model('mnist_cnn_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.json['image']
    image_data = re.sub('^data:image/.+;base64,', '', image_data)

    raw = Image.open(io.BytesIO(base64.b64decode(image_data)))
    #raw.save("debug_stage_1_raw_from_canvas.png")

    # Convert to grayscale
    image = raw.convert('L')
    #image.save("debug_stage_2_grayscale.png")

    # Invert image: 
    # white background -> black, 
    # black digits -> white

    image = Image.eval(image, lambda x: 255 - x)
    #image.save("debug_stage_3_inverted.png")

    image = image.resize((28, 28))

    image_np = np.array(image) / 255.0
    image_np = image_np.reshape(1, 28, 28, 1)

    Image.fromarray((image_np[0, :, :, 0] * 255).astype('uint8'))

    prediction = model.predict(image_np)
    predicted_digit = int(np.argmax(prediction))

    return jsonify({'prediction': predicted_digit})


if __name__ == '__main__':
    app.run(debug=True)
