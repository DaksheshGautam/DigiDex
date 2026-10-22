# GlyphDex — Handwriting recognition

Draw a short line of **separate digits, uppercase/lowercase letters and supported symbols**. GlyphDex splits the drawing into characters and reads them from left to right. Choose mixed, digits, uppercase, lowercase or symbols mode. Uncertain readings show `?`; confirm a suggestion or enter the correct character before copying. Undo strokes and copy the result.

Supported: `0–9`, `A–Z`, `a–z`, and `+ - / < > # $ % & [ ] { } × ÷ ± ≤ ≥ ≠ ≈ =` (83 classes). The equals sign uses generated two-stroke training examples and is experimental. `!`, `?`, `@`, commas, parentheses and other unlisted punctuation are not trained classes.

Write one spaced line with up to 32 characters. Dots and stacked strokes remain together when their horizontal ranges overlap; a small, slightly offset dot above a neighbouring stem is also merged. Joined cursive, overlapping characters, multiple lines and badly separated strokes remain unsupported. Spaces are estimated from larger gaps. O/0, I/l/1, and letter case can be visually ambiguous. Rejection uses conservative score, margin and shape rules, plus known mixed-mode ambiguities; it is not calibrated and cannot detect every unsupported input or confidently wrong answer.

**Live app:** [digidex-dakshesh.onrender.com](https://digidex-dakshesh.onrender.com/)

## Trained model

The character model was trained in the owner's logged-in **Google Colab T4 GPU** session. Full Keras and equivalent NumPy weights for both the new model and retained baseline are committed in [`models/characters/`](models/characters/), with labels, measured metrics, verification examples and a model card. Training and inference share [`character_preprocessing.py`](character_preprocessing.py).

Held-out accuracy by recognition mode: digits **99.40%**, uppercase **97.67%**, lowercase **93.28%**, symbols **98.39%**. Each result measures classification before uncertainty checks and assumes the correct focused mode is selected; these are individual-character results, not mixed-mode or complete-line accuracy. Synthetic `=` is excluded. See the model card for sample counts and limitations.

The original training run is preserved in [`GlyphDex_Characters_Colab.ipynb`](GlyphDex_Characters_Colab.ipynb). The new [`GlyphDex_v2_Colab.ipynb`](GlyphDex_v2_Colab.ipynb) has 12 numbered code sections for GPU setup, downloads, data preparation, stroke augmentation, training, separate mode evaluation, class-balanced fine-tuning, baseline comparison and export. Use **Runtime → Run all** on a free Colab T4. It uses more EMNIST examples and varied stroke thickness. The released model trained for 15 initial epochs and 9 class-balanced fine-tuning epochs. Focused modes use v2. Mixed mode uses 25% v2 and 75% baseline probabilities, selected on shared held-out EMNIST validation with a category regression gate. The model card records the mixed-mode tradeoffs; improvements are modest and real mouse-drawing accuracy is not yet established. Training requires TensorFlow; the deployed app only needs NumPy, Pillow, Flask, Gunicorn and Psycopg for optional private storage. Complete-line and real website accuracy require their own held-out evaluation.

### Corrected website examples

Connect the **Private online dataset** once using your owner key. With auto-save enabled, choosing an alternative or entering a corrected character automatically queues its crop, label, original stroke coordinates and drawing ID. Enter a corrected full line and press Enter or leave the field to save the original canvas for segmentation evaluation. Predictions alone never become training labels. Each character box must contain exactly that character; use the line field when boxes are wrong.

Corrections stay private in PostgreSQL. Only authenticated owners can save or export them. Keys are not stored in browser storage: the browser uses a Secure, HttpOnly, SameSite session cookie and CSRF token. Pending corrections survive refreshes in a local queue and retry on connection or **Sync now**. Later labels replace earlier labels for the same normalized drawing. Keep each real person's writer ID consistent across devices. Disconnect or pause auto-save at any time. **Export online dataset** downloads a private backup; **Export queued corrections** backs up unsynced data. Existing **Save checked examples / Export examples** controls still support browser-only collection.

Open [`GlyphDex_Private_Dataset_Colab.ipynb`](GlyphDex_Private_Dataset_Colab.ipynb) in your private Colab session to download the dataset with a hidden key prompt and prepare `website_corrections.npz`. Keep the JSON, NPZ, access key and executed notebook private. Nothing is uploaded to public GitHub. This notebook prepares data only; retraining remains a separate action.

For one person's handwriting, select `drawing` splits: all crops from one canvas stay together across train, validation and test. This evaluates personalization on held-out drawings, not unseen writers. At least three distinct drawings are required; substantially more diverse examples are needed for useful improvement. For a general handwriting benchmark, collect at least three real writers and use `writer` splits. Never invent writer IDs to satisfy that requirement. Conflicting duplicate labels are rejected.

Upload the prepared `website_corrections.npz` through the training Colab Files panel before running the v2 notebook. Section 6 adds only training and validation examples. Section 10 evaluates separate website test examples by recognition mode. Keep the split report with the trained model so personalization results are not presented as unseen-writer accuracy. Without that file, training uses only public datasets. Collect varied examples and inspect labels before retraining; collecting corrections does not automatically change the live model.

Full lines evaluate the complete prediction pipeline separately. Run `python evaluate_lines.py glyphdex-private-dataset.json --split-report website_corrections.json`. The report selects held-out writers or drawings and counts rejected characters, wrong spaces and request errors as failures. Omitting the split report is exploratory. Full lines never automatically enter classifier training.

The original **MNIST-only** model remains in [`models/`](models/) for compatibility with requests that omit `mode`:

- [`mnist_cnn.keras`](models/mnist_cnn.keras): complete Keras model.
- [`mnist_cnn_weights.npz`](models/mnist_cnn_weights.npz): equivalent weights used by the lightweight NumPy web runtime.
- [`metrics.json`](models/metrics.json): measured results; [`MODEL_CARD.md`](models/MODEL_CARD.md): training details and limitations.

Its original training notebook is [`DigiDex_Colab_Training.ipynb`](DigiDex_Colab_Training.ipynb); its test accuracy is 98.86% on 10,000 MNIST test digits.

## Run locally

```bash
pip install -r requirements.txt
python app.py
python -m unittest discover -v
node test_dataset_sync.js
```

Open http://127.0.0.1:5000. Tests compare NumPy inference with Colab Keras outputs for both exports, check mixed-character HTTP recognition, dotted/stacked glyph segmentation, blank drawings, limits and malformed input. Inference processes at most four characters at once to bound memory on Render Free.

### Prediction API

`POST /predict` accepts `{"image":"data:image/png;base64,...","mode":"mixed"}`. The canvas is white ink on black. Supported modes: `mixed`, `digits`, `uppercase`, `lowercase`, `symbols`. The response includes `text`, ordered `characters`, `suggestion`, `uncertain`, `reasons`, canvas-coordinate bounding boxes, model scores and three alternatives. Rejected characters use `?`, which is a placeholder rather than a trained punctuation class. Omitting `mode` retains the original single-digit response. Images are capped at 1024×1024 and 2 MB per request. Server predictions are processed in memory and are not stored; explicit corrections are saved only when the owner enables private collection; browser-only exports remain available.

## Render deployment

Create a Python web service from this repository on the Free plan. Build: `pip install -r requirements.txt`. Start: `gunicorn --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT app:app`. Health check: `/healthz`. Python 3.12.14 and `OPENBLAS_NUM_THREADS=1`. These settings are also in `render.yaml`. Free services sleep when idle; the first request may take longer. For a public-repository service, use Render's manual deploy control after GitHub updates unless automatic deployment is configured.

### Private dataset configuration

Configure `GLYPHDEX_DATABASE_URL` with a dedicated restricted PostgreSQL role, `GLYPHDEX_DATABASE_CA` with the provider CA certificate, `GLYPHDEX_DATASET_KEY_HASH` with SHA-256 of a random owner key, and `GLYPHDEX_DATASET_SESSION_SECRET` with a separate random session secret. TLS verifies the database hostname and CA. Keep values in Render secret environment settings, never source control. A dedicated database owner runs `DatasetStore.initialize()` once, then grants the web role only schema USAGE and table SELECT, INSERT, UPDATE. The web app cannot create tables or delete examples. Without configuration, recognition and local exports still work.

`POST /dataset/session` accepts an owner key and creates the browser session. Cookie-authenticated writes require `X-Dataset-CSRF` and the same HTTPS origin. Colab uses `Authorization: Bearer <owner-key>` for `GET /dataset/export?limit=200&cursor=...`. Exports are private and uncached. Each page includes `examples`, `lines`, and `next_cursor`; follow pages until the cursor is null. No key or database URI appears in API error messages.

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


