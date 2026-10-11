"""Small NumPy runtime for the trained Keras CNN."""
from pathlib import Path
import numpy as np
from PIL import Image

def conv(x, kernel, bias):
    windows = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(1, 2))
    return np.maximum(np.einsum('nhwcij,ijco->nhwo', windows, kernel, optimize=True) + bias, 0)

def pool(x):
    n, h, w, c = x.shape
    return x[:, :h//2*2, :w//2*2].reshape(n, h//2, 2, w//2, 2, c).max(axis=(2, 4))

class DigitModel:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.weights = [data[f'w{i}'] for i in range(8)]

    def predict(self, images):
        w = self.weights
        x = np.asarray(images, dtype=np.float32)
        x = pool(conv(x, w[0], w[1]))
        x = pool(conv(x, w[2], w[3])).reshape(len(x), -1)
        x = np.maximum(x @ w[4] + w[5], 0)
        x = x @ w[6] + w[7]
        x = np.exp(x - x.max(axis=1, keepdims=True))
        return x / x.sum(axis=1, keepdims=True)

def prepare(image):
    image = image.convert('L')
    values = np.asarray(image)
    ys, xs = np.where(values > 25)
    if len(xs) < 10:
        raise ValueError('Draw a digit first.')
    image = image.crop((int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1))
    scale = 20 / max(image.size)
    image = image.resize(tuple(max(1, round(s*scale)) for s in image.size), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (28, 28))
    canvas.paste(image, ((28-image.width)//2, (28-image.height)//2))
    array = np.asarray(canvas, dtype=np.float32)
    yy, xx = np.indices(array.shape)
    shift_x = round(13.5 - float((xx*array).sum()/array.sum()))
    shift_y = round(13.5 - float((yy*array).sum()/array.sum()))
    centered = Image.new('L', (28,28))
    centered.paste(canvas, (shift_x, shift_y))
    return np.asarray(centered, dtype=np.float32)[None, ..., None] / 255.0
