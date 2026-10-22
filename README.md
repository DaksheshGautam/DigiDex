# GlyphDex — Handwriting recognition

Draw a short line of **separate digits, uppercase/lowercase letters and supported symbols**. GlyphDex splits the drawing into characters and reads them from left to right. Choose mixed, digits, uppercase, lowercase or symbols mode. Uncertain readings show `?`; confirm a suggestion or enter the correct character before copying. Undo strokes and copy the result.

Supported: `0–9`, `A–Z`, `a–z`, and `+ - / < > # $ % & [ ] { } × ÷ ± ≤ ≥ ≠ ≈ =` (83 classes). The equals sign uses generated two-stroke training examples and is experimental. `!`, `?`, `@`, commas, parentheses and other unlisted punctuation are not trained classes.

Write one spaced line with up to 32 characters. Dots and stacked strokes remain together when their horizontal ranges overlap; a small, slightly offset dot above a neighbouring stem is also merged. Joined cursive, overlapping characters, multiple lines and badly separated strokes remain unsupported. Spaces are estimated from larger gaps. O/0, I/l/1, and letter case can be visually ambiguous. Rejection uses conservative score, margin and shape rules, plus known mixed-mode ambiguities; it is not calibrated and cannot detect every unsupported input or confidently wrong answer.

**Live app:** [digidex-dakshesh.onrender.com](https://digidex-dakshesh.onrender.com/)

## Trained model

The character model was trained in the owner's logged-in **Google Colab T4 GPU** session. Full Keras and equivalent NumPy weights for both the new model and retained baseline are committed in [`models/characters/`](models/characters/), with labels, measured metrics, verification examples and a model card. Training and inference share [`character_preprocessing.py`](character_preprocessing.py).

Held-out accuracy by recognition mode: digits **99.40%**, uppercase **97.67%**, lowercase **93.28%**, symbols **98.39%**. Each result measures classification before uncertainty checks and assumes the correct focused mode is selected; these are individual-character results, not mixed-mode or complete-line accuracy. Synthetic `=` is excluded. See the model card for sample counts and limitations.

The original training run is preserved in [`GlyphDex_Characters_Colab.ipynb`](GlyphDex_Characters_Colab.ipynb). The new [`GlyphDex_v2_Colab.ipynb`](GlyphDex_v2_Colab.ipynb) has 12 numbered code sections for GPU setup, downloads, data preparation, stroke augmentation, training, separate mode evaluation, class-balanced fine-tuning, baseline comparison and export. Use **Runtime → Run all** on a free Colab T4. It uses more EMNIST examples and varied stroke thickness. The released model trained for 15 initial epochs and 9 class-balanced fine-tuning epochs. Focused modes use v2. Mixed mode uses 25% v2 and 75% baseline probabilities, selected on shared held-out EMNIST validation with a category regression gate. The model card records the mixed-mode tradeoffs; improvements are modest and real mouse-drawing accuracy is not yet established. Training requires TensorFlow; the deployed app only needs NumPy, Pillow, Flask and Gunicorn. Complete-line and real website accuracy require their own held-out evaluation.

### Corrected website examples

After checking every character and bounding box, choose **Save checked examples**. Examples are stored only in that browser's local storage. **Export examples** downloads a JSON file; no automatic upload or retraining takes place. Export before clearing browser data. Draw one character at a time if the segmentation boxes are wrong; a label correction cannot fix a crop containing two characters.

Validate exports with `python correction_data.py writer1.json writer2.json writer3.json`. It produces `website_corrections.npz` and a split report. At least three distinct real writers are required: each writer stays entirely in train, validation or test. Browser group identifiers are only a starting point; merge groups from the same person and separate exports for different people before preparing the dataset. Exact duplicate images with conflicting labels or writer groups are rejected. These exports are not committed to the public repository. No real website accuracy is claimed until those independent examples exist and are evaluated.

Upload the validated `website_corrections.npz` through Colab's Files panel **before** running the notebook. Section 6 adds only its training and validation groups to those splits. Section 10 evaluates its separate website test group by recognition mode. Without that file, the notebook trains only on the public datasets and reports zero real website test examples.

For segmentation mistakes, enter **Correct full line** and choose **Save line for testing**. This saves the original canvas and independent expected text, even if boxes or suggestions are wrong. Exports include those lines alongside character crops. Run `python evaluate_lines.py export.json --split-report website_corrections.json` to check only test writers through the complete prediction pipeline. It reports exact-line results separately by selected mode, including rejected characters, incorrect spaces and request errors as failures. Omitting the split report is an exploratory check, not a held-out benchmark. Line examples never automatically enter model training.

The original **MNIST-only** model remains in [`models/`](models/) for compatibility with requests that omit `mode`:

- [`mnist_cnn.keras`](models/mnist_cnn.keras): complete Keras model.
- [`mnist_cnn_weights.npz`](models/mnist_cnn_weights.npz): equivalent weights used by the lightweight NumPy web runtime.
- [`metrics.json`](models/metrics.json): measured results; [`MODEL_CARD.md`](models/MODEL_CARD.md): training details and limitations.

Its original training notebook is [`DigiDex_Colab_Training.ipynb`](DigiDex_Colab_Training.ipynb); its test accuracy is 98.86% on 10,000 MNIST test digits.

## Run locally

```bash
pip install -r requirements.txt
python app.py
python -m unittest -v test_app test_characters test_quality
```

Open http://127.0.0.1:5000. Tests compare NumPy inference with Colab Keras outputs for both exports, check mixed-character HTTP recognition, dotted/stacked glyph segmentation, blank drawings, limits and malformed input. Inference processes at most four characters at once to bound memory on Render Free.

### Prediction API

`POST /predict` accepts `{"image":"data:image/png;base64,...","mode":"mixed"}`. The canvas is white ink on black. Supported modes: `mixed`, `digits`, `uppercase`, `lowercase`, `symbols`. The response includes `text`, ordered `characters`, `suggestion`, `uncertain`, `reasons`, canvas-coordinate bounding boxes, model scores and three alternatives. Rejected characters use `?`, which is a placeholder rather than a trained punctuation class. Omitting `mode` retains the original single-digit response. Images are capped at 1024×1024 and 2 MB per request. Server predictions are processed in memory and are not stored; optional checked examples remain in the browser until explicitly exported.

## Render deployment

Create a Python web service from this repository on the Free plan. Build: `pip install -r requirements.txt`. Start: `gunicorn --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT app:app`. Health check: `/healthz`. Python 3.12.14 and `OPENBLAS_NUM_THREADS=1`. These settings are also in `render.yaml`. Free services sleep when idle; the first request may take longer. For a public-repository service, use Render's manual deploy control after GitHub updates unless automatic deployment is configured.

## Original project

The original research notebooks and project description are preserved below.

# 👩‍💻 DigiDex
# Determining Digits Smartly

# 🔢 Building an ML Model for digit recognition

🤖 In this project, we have created an ML model that uses CNN classification algorithm to read digits and identify them. We have worked upon a standard MNIST dataset of thousands of digits. By using this dataset, we have trained, tested and validated our model.

We have tested various algorithms and compared their accuracy, precision, F1 score and recall to generate the most effective model.

📊 Here are the comparisons of various metrics and other relevant plots:


![Screenshot 2023-03-25 214810](https://user-images.githubusercontent.com/122363068/227737713-5ac6843e-c9aa-4983-8af1-f460cb452888.png)

![Screenshot 2023-03-25 225914](https://user-images.githubusercontent.com/122363068/227737759-1b6bf77b-8fbe-45db-bd46-a71393c961c0.png)

![Screenshot 2023-03-25 225937](https://user-images.githubusercontent.com/122363068/227737792-9e7dc5a8-afdb-45dc-b744-44b5067403c4.png)


Through this project we got to learn a lot about Machine Learning and AI Models, we trained our model on large data sets, implemented their splitting and formulated an organised and effective code.


💻 The tech stack we used:

1. Python
2. Libraries: matplotlib, pandas, keras, scikit, TensorFlow etc.
3. Classification algorithms: CNN, Randon Forest,SVM, KNN, DBN, MLP.
4. Google Collab


🖋 END NOTE:
During the course of this project, we got to comprehend various research articles which supplemented us in writing an optimised code and intricately understanding the mechanism of machine learning.


