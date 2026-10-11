# DigiDex web app

Draw a single handwritten digit (0–9) and get a CNN prediction with probabilities for all ten classes. The model was trained in Google Colab and achieved **98.86% accuracy on 10,000 held-out MNIST test images**.

## Trained model

Both model formats are committed in [`models/`](models/):

- [`mnist_cnn.keras`](models/mnist_cnn.keras): complete Keras model.
- [`mnist_cnn_weights.npz`](models/mnist_cnn_weights.npz): equivalent weights used by the lightweight NumPy web runtime.
- [`metrics.json`](models/metrics.json): measured results; [`MODEL_CARD.md`](models/MODEL_CARD.md): training details and limitations.

Retrain using [`DigiDex_Colab_Training.ipynb`](DigiDex_Colab_Training.ipynb) in Colab. Run the training cell, then download its artifact ZIP. Training requires TensorFlow; the deployed app only needs NumPy, Pillow, Flask and Gunicorn.

## Run locally

```bash
pip install -r requirements.txt
python app.py
python -m unittest -v test_app
```

Open http://127.0.0.1:5000. Tests compare the runtime against 100 original Colab Keras predictions and check the HTTP prediction flow and input errors.

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


