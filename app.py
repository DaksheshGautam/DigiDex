import base64
import io
import json
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from inference import DigitModel, prepare

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
model = DigitModel(ROOT / 'models/mnist_cnn_weights.npz')
metrics = json.loads((ROOT / 'models/metrics.json').read_text())

@app.get('/')
def index():
    return render_template('index.html', accuracy=f"{metrics['test_accuracy']*100:.2f}")

@app.get('/healthz')
def health():
    return jsonify(status='ok', model='MNIST CNN')

@app.post('/predict')
def predict():
    try:
        payload = request.get_json()
        encoded = payload['image'].split(',', 1)[1]
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.width > 1024 or image.height > 1024:
                raise ValueError('Image is too large.')
            probabilities = model.predict(prepare(image))[0]
        return jsonify(digit=int(probabilities.argmax()), probabilities=probabilities.tolist())
    except (ValueError, KeyError, IndexError, TypeError, UnidentifiedImageError, OSError):
        return jsonify(error='Draw one digit on the canvas and try again.'), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
