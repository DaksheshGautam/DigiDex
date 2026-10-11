"""Run in a Google Colab code cell; training happens in its hosted runtime."""
import json
import zipfile
from pathlib import Path
import numpy as np
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32')[..., None] / 255.0
x_test = x_test.astype('float32')[..., None] / 255.0
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train[:55000], y_train[:55000], validation_data=(x_train[55000:], y_train[55000:]),
                    epochs=5, batch_size=128, verbose=2)
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
out = Path('digidex_model')
out.mkdir(exist_ok=True)
model.save(out / 'mnist_cnn.keras')
np.savez_compressed(out / 'mnist_cnn_weights.npz', **{f'w{i}': w for i, w in enumerate(model.get_weights())})
reference = model.predict(x_test[:100], verbose=0)
np.savez_compressed(out / 'verification.npz', images=x_test[:100], labels=y_test[:100], probabilities=reference)
metrics = {'dataset': 'MNIST', 'seed': 42, 'train_examples': 55000, 'validation_examples': 5000,
           'test_examples': 10000, 'epochs': 5, 'batch_size': 128, 'test_loss': float(loss),
           'test_accuracy': float(accuracy), 'tensorflow_version': tf.__version__,
           'training_environment': 'Google Colab', 'history': history.history}
(out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
with zipfile.ZipFile('digidex_model.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for file in out.iterdir():
        archive.write(file, file.name)
print('TEST_ACCURACY:', accuracy)
print('READY: digidex_model.zip')
