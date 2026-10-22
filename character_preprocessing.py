"""Shared glyph normalization for Colab training and the web app."""
import numpy as np
from PIL import Image


def normalize_glyph(image):
    image = image.convert('L')
    values = np.asarray(image)
    ys, xs = np.where(values > 25)
    if len(xs) < 3:
        raise ValueError('Draw a character first.')
    image = image.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    scale = 20 / max(image.size)
    image = image.resize(tuple(max(1, round(s * scale)) for s in image.size), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (28, 28))
    canvas.paste(image, ((28 - image.width) // 2, (28 - image.height) // 2))
    array = np.asarray(canvas, dtype=np.float32)
    yy, xx = np.indices(array.shape)
    shift_x = round(13.5 - float((xx * array).sum() / array.sum()))
    shift_y = round(13.5 - float((yy * array).sum() / array.sum()))
    centered = Image.new('L', (28, 28))
    centered.paste(canvas, (shift_x, shift_y))
    return np.asarray(centered, dtype=np.float32)[..., None] / 255.0


def segment_line(image, max_characters=32):
    """Split one spaced line by ink columns, retaining dots and stacked strokes."""
    image = image.convert('L')
    ink = np.asarray(image) > 25
    columns = np.flatnonzero(ink.any(axis=0))
    if not len(columns):
        raise ValueError('Draw some characters first.')
    # A two-pixel break can be an antialiasing gap inside one character.
    groups = np.split(columns, np.flatnonzero(np.diff(columns) > 3) + 1)
    boxes, glyphs = [], []
    for group in groups:
        left, right = int(group[0]), int(group[-1]) + 1
        ys = np.flatnonzero(ink[:, left:right].any(axis=1))
        top, bottom = int(ys[0]), int(ys[-1]) + 1
        if ink[top:bottom, left:right].sum() < 3:
            continue
        boxes.append([left, top, right, bottom])
    # A slightly offset i/j dot can occupy different columns from its stem.
    # Merge only a small detached mark above a much taller neighbouring stroke;
    # ordinary adjacent letters and horizontal symbol strokes remain separate.
    merged = True
    while merged:
        merged = False
        for i, dot in enumerate(boxes):
            dw, dh = dot[2]-dot[0], dot[3]-dot[1]
            for j, stem in enumerate(boxes):
                if i == j:
                    continue
                sh = stem[3]-stem[1]
                gap = max(stem[0]-dot[2], dot[0]-stem[2], 0)
                if (dh <= sh*.28 and dw <= sh*.23 and dot[3] < stem[1]
                        and stem[1]-dot[3] <= sh*.55 and gap <= sh*.16):
                    boxes[j] = [min(dot[0],stem[0]),dot[1],max(dot[2],stem[2]),stem[3]]
                    boxes.pop(i)
                    merged = True
                    break
            if merged:
                break
    boxes.sort(key=lambda b:b[0])
    glyphs = [normalize_glyph(image.crop(box)) for box in boxes]
    if not glyphs:
        raise ValueError('Draw some characters first.')
    if len(glyphs) > max_characters:
        raise ValueError(f'Please draw at most {max_characters} characters on one line.')
    return np.stack(glyphs), boxes
