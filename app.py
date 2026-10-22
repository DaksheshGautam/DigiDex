import base64
import io
import json
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
import numpy as np
from inference import DigitModel, prepare
from character_preprocessing import segment_line
from recognition_quality import assess

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
model = DigitModel(ROOT / 'models/mnist_cnn_weights.npz')
metrics = json.loads((ROOT / 'models/metrics.json').read_text())
character_model = DigitModel(ROOT / 'models/characters/characters_cnn_weights.npz')
labels = json.loads((ROOT / 'models/characters/labels.json').read_text(encoding='utf-8'))
character_metrics = json.loads((ROOT / 'models/characters/metrics.json').read_text(encoding='utf-8'))
mixed_candidate_weight = float(character_metrics.get('routing',{}).get('mixed_candidate_weight',1.0))
baseline_model = (DigitModel(ROOT / 'models/characters/baseline_cnn_weights.npz')
                  if mixed_candidate_weight < 1 else None)

@app.get('/')
def index():
    return render_template('characters.html',
                           test_examples=character_metrics['test_examples'], symbols=' '.join(labels[62:]),
                           mode_scores={k:f"{v['accuracy']*100:.1f}" for k,v in character_metrics['mode_accuracy'].items()})

@app.get('/healthz')
def health():
    return jsonify(status='ok', model='GlyphDex character CNN', classes=len(labels),
                   version=character_metrics.get('version','v1'), rejection=True,
                   mixed_candidate_weight=mixed_candidate_weight)

@app.post('/predict')
def predict():
    try:
        payload = request.get_json()
        if not isinstance(payload, dict) or not isinstance(payload.get('image'), str):
            raise ValueError('Draw some characters and try again.')
        encoded = payload['image'].split(',', 1)[1]
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.width > 1024 or image.height > 1024:
                raise ValueError('Image is too large.')
            if payload.get('mode'):
                return predict_line(image, payload['mode'])
            probabilities = model.predict(prepare(image))[0]
        return jsonify(digit=int(probabilities.argmax()), probabilities=probabilities.tolist())
    except (ValueError, KeyError, IndexError, TypeError, UnidentifiedImageError, OSError):
        return jsonify(error='Draw on the canvas and try again. Use an image no larger than 1024 × 1024.'), 400


@app.errorhandler(413)
def oversized(_error):
    return jsonify(error='The drawing is too large. Please keep the image under 2 MB.'), 413


def character_probabilities(glyphs, mode):
    batches=[]
    for i in range(0,len(glyphs),4):
        images=glyphs[i:i+4]
        if mode=='mixed' and baseline_model is not None:
            row=baseline_model.predict(images)*(1-mixed_candidate_weight)
            if mixed_candidate_weight:
                row+=character_model.predict(images)*mixed_candidate_weight
        else:
            row=character_model.predict(images)
        batches.append(row)
    return np.concatenate(batches)


def predict_line(image, mode):
    if mode not in ('mixed', 'digits', 'uppercase', 'lowercase', 'symbols'):
        return jsonify(error='Choose a valid recognition mode.'), 400
    try:
        glyphs, boxes = segment_line(image)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    # Bound convolution working memory on Render's free 512 MB service.
    probabilities = character_probabilities(glyphs, mode)
    allowed = {'mixed': np.arange(len(labels)), 'digits': np.arange(10),
               'uppercase': np.arange(10,36), 'lowercase': np.arange(36,62),
               'symbols': np.arange(62,len(labels))}[mode]
    characters = []
    text = ''
    typical_width = max(float(np.median([b[2]-b[0] for b in boxes])), 12)
    for i, (row, box) in enumerate(zip(probabilities, boxes)):
        ranking = allowed[np.argsort(row[allowed])[::-1][:3]]
        best = int(ranking[0])
        space = bool(i and box[0]-boxes[i-1][2] > max(typical_width*.95, image.height*.22))
        if space:
            text += ' '
        reasons = assess(row, ranking, labels, mode, glyphs[i])
        chosen = '?' if reasons else labels[best]
        text += chosen
        characters.append({'character':chosen, 'suggestion':labels[best], 'uncertain':bool(reasons),
                           'reasons':reasons, 'confidence':float(row[best]), 'box':box,
                           'space_before':space, 'experimental':labels[best]=='=',
                           'alternatives':[{'character':labels[int(j)],'confidence':float(row[j])} for j in ranking]})
    return jsonify(text=text, characters=characters, mode=mode)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
