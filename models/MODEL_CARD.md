# DigiDex CNN

Trained in the account owner's Google Colab T4 GPU runtime on October 11, 2026.

- Dataset: MNIST handwritten digits, labels 0–9.
- Split: 55,000 training, 5,000 validation, 10,000 held-out test images.
- Architecture: Conv2D(32, 3) → max pool → Conv2D(64, 3) → max pool → flatten → Dense(128) → Dense(10, softmax). ReLU hidden activations.
- Adam, sparse categorical cross entropy, 5 epochs, batch size 128, seed 42.
- Held-out test accuracy: **98.86%**; test loss: 0.03267648.
- TensorFlow version in Colab: 2.21.0.

`mnist_cnn.keras` contains the complete trainable model and optimizer state. `mnist_cnn_weights.npz` contains the same learned weights for lightweight NumPy inference on Render. `metrics.json` contains measured training and test results. `verification.npz` contains 100 MNIST test examples and their original Keras predictions. The NumPy runtime matched all 100 predicted labels, with maximum probability difference 2.98e-7.

The app crops white canvas ink, scales it into a 20×20 box, and centers it in a 28×28 image. Accuracy measured on MNIST does not guarantee the same accuracy on mouse drawings. This model recognizes single digits only; letters, multi-digit numbers and unusual drawings are outside its intended use. Softmax confidence is not calibrated.

MNIST dataset attribution: Yann LeCun and Corinna Cortes, [MNIST](https://yann.lecun.org/exdb/mnist/), distributed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). The 100 verification images are resized in representation only and remain under that license. Original project notebooks are retained.
