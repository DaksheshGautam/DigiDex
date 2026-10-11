# GlyphDex character CNN

83 output classes: 10 digits, 26 uppercase letters, 26 lowercase letters, 20 handwritten symbols, and experimental `=`. Trained by Dakshesh Gautam in his logged-in Google Colab T4 GPU runtime on October 11, 2026. No local model training.

Held-out accuracy by recognition mode: digits **99.20%** (3,000), uppercase **97.22%** (7,800), lowercase **92.83%** (7,800), symbols **98.08%** (1,616). These metrics use the same model and held-out predictions, restricting candidate output classes to the stated group; selecting a mode for an incompatible drawing does not have this guarantee. They are not mixed-mode or complete-line accuracies. Detailed per-class and category measurements are preserved in the metrics file.

The equivalent NumPy runtime matched all 246 exported Keras predicted labels spanning the 82 handwritten classes. Maximum absolute probability difference: **1.97e-6**. Six automated tests passed, including mixed `A7+` recognition, segmentation of dotted and stacked glyphs, input errors and legacy MNIST inference.

## Data and evaluation

- EMNIST ByClass: random seed 42, at most 2,700 official training images per class, 10% of those reserved for validation; at most 300 official test images per class. Images are transposed to upright orientation. Classes with fewer available examples use all available examples.
- HASYv2: 20 selected symbol classes; the official fold-1 test images are held out. 10% (at least five) of the remaining images per class are reserved for validation. Only the training subset is sampled with replacement to 2,430 presentations per symbol. Repeated presentations are not new unique handwritten samples. This is a single-fold subset experiment, not the published 10-fold HASY benchmark. HASY IDs do not reliably identify individual writers, so this is not a writer-disjoint evaluation.
- `=`: 2,430 procedural two-stroke training images, 270 procedural validation images. No real handwritten equals test set; this class is experimental and excluded from all reported test metrics.
- Exact counts, epochs, history, per-class, category and focused-mode metrics are in [`metrics.json`](metrics.json). Class sampling means these results do not estimate the original dataset's natural-frequency accuracy.

Preprocessing is shared with the app: crop ink above intensity 25, preserve aspect ratio while fitting into a 20×20 region, paste into a 28×28 canvas and center by ink mass. HASY black-on-white images are inverted. Training augmentation applies small rotations and translations without horizontal flips.

This run used 199,619 training presentations, 18,209 validation examples and 12 epochs. The entire training set is shuffled each epoch; smaller buffers are unsuitable because source images are collected in class order.

Architecture: Conv2D(32, 3, ReLU) → max pool → Conv2D(64, 3, ReLU) → max pool → flatten → Dense(128, ReLU) → Dropout(0.2) → Dense(83, softmax). Adam, sparse categorical cross-entropy, batch size 256, up to 12 epochs, validation-loss early stopping with best weights restored, and learning-rate reduction. Dropout is disabled during inference.

## Artifacts

- `characters_cnn.keras`: complete Keras model with optimizer state.
- `characters_cnn_weights.npz`: eight learned arrays for the equivalent NumPy CNN.
- `labels.json`: ordered class labels; do not reorder without retraining.
- `verification.npz`: up to three held-out glyphs per handwritten class with original Colab Keras probabilities, used solely to verify exported inference. It is not the complete test set.
- Training code: [`train_characters_colab.py`](../../train_characters_colab.py); reproducible browser notebook: [`GlyphDex_Characters_Colab.ipynb`](../../GlyphDex_Characters_Colab.ipynb).

## Intended use and limitations

One line of separate characters on the app's white-on-black drawing canvas. The app splits groups of ink columns (merging breaks up to two empty pixels), keeps horizontally overlapping disconnected strokes together and reads left to right. Groups may contain multiple touching characters or separate parts of one badly spaced character. Bounding boxes and alternatives help users inspect these errors. Up to 32 characters per drawing; inference batches of four bound memory on Render Free.

O/0, I/l/1, some upper/lowercase pairs and x/× can be visually indistinguishable. Modes narrow labels but confidence remains the unrestricted model's probability, not a recalibrated guarantee. Softmax scores are not calibrated. No language model or word correction is used. Larger gaps heuristically produce spaces.

Joined cursive, overlapping characters, multiple lines, document/photo OCR, arbitrary punctuation and mathematical expression layout are outside this release. `!`, `?`, `@`, commas and parentheses are not classes. The individual-glyph test accuracy is not whole-line accuracy and does not guarantee accuracy on mouse drawings.

## Sources and attribution

EMNIST: Gregory Cohen, Saeed Afshar, Jonathan Tapson and André van Schaik, [EMNIST: an extension of MNIST to handwritten letters](https://arxiv.org/abs/1702.05373), 2017. Original dataset: [NIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) / [Western Sydney University](https://rds.westernsydney.edu.au/Institutes/MARCS/BENS/EMNIST/). Colab rejected those download endpoints with HTTP 403; this run obtained the original IDX gzip format from the [Heliosoph EMNIST data mirror](https://huggingface.co/datasets/Heliosoph/EMNIST). Upstream NIST data originates from NIST Special Database 19. Mirror metadata lists CC BY 4.0; preserve attribution for redistributed verification samples.

HASYv2: Martin Thoma, [The HASYv2 dataset](https://arxiv.org/abs/1701.08380), 2017, [Zenodo release](https://doi.org/10.5281/zenodo.259444), MD5 `fddf23f36e24b5236f6b3a0880c778e3`. Derived from Detexify (Daniel Kirsch) and write-math recordings. The dataset is published under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/). HASY-derived verification glyphs retain that source attribution and database license; the full source database is available at the linked release. Original HASY glyphs were inverted, cropped, resized and centered here. Procedural equals examples were generated by this training script. Original DigiDex notebooks remain preserved separately.
