# GlyphDex character recognition v2

Trained by Dakshesh Gautam in his logged-in Google Colab T4 GPU session on October 22, 2026. All model fitting happened in Colab. There are 83 classes: digits, uppercase/lowercase English letters, 20 handwritten symbols and experimental `=`.

## Focused-mode classification results

Both models were evaluated on the **same** held-out individual characters. These scores are before uncertainty filtering; the correct focused mode must be selected. They are not complete-line or real website accuracy.

| Mode | Test glyphs | Baseline | v2 |
|---|---:|---:|---:|
| Digits | 3,000 | 99.20% | 99.40% |
| Uppercase | 7,800 | 97.22% | 97.67% |
| Lowercase | 7,800 | 92.83% | 93.28% |
| Symbols | 1,616 | 98.08% | 98.39% |

The gains are modest. No independent user-drawn website benchmark has been collected. Synthetic equals examples are excluded from the handwritten test metrics.

## Mixed-mode model selection and tradeoffs

The new model alone regressed on mixed lowercase recognition, so it is not used alone for mixed mode. The deployed runtime averages probabilities with **25% v2 + 75% baseline**. Candidate weights 0, 0.25, 0.5, 0.75 and 1 were compared on the shared held-out EMNIST validation prefix, excluding the extra new validation examples that the baseline had trained on. HASY validation examples were excluded from blending selection because its resampling changed the validation subsets. Selection maximized mean per-class validation correctness subject to a 0.5-point category regression limit. A separate held-out category regression check can fall back to the baseline. Model-selection use of validation and test gates must be considered when interpreting the reported comparisons.

| Mixed-mode input group | Baseline | Deployed blend |
|---|---:|---:|
| Digits | 84.43% | 85.77% |
| Uppercase | 77.96% | 79.22% |
| Lowercase | 66.60% | 66.21% |
| Symbols | 97.90% | 98.02% |

Mixed lowercase decreased by 0.40 points; mixed mode is not uniformly improved. O/0, I/l/1, upper/lowercase and x/× remain ambiguous. Use a focused mode whenever possible. Full trial details and the rejected unblended candidate's results remain in `metrics.json`.

## Data and training

- EMNIST ByClass: seed 42; cap 6,000 official training images per class, 10% reserved for validation; cap 300 official test images per class. Upright orientation is restored by transposing images. Normalization crops ink, preserves aspect ratio in a 20-pixel region, pads to 28×28 and centers by ink mass.
- HASYv2: the same 20 selected symbols and official fold-1 test split. Hold out 10% (at least five) of available training images for validation. Only training images are resampled to 5,400 presentations per symbol. Repeated presentations are not new unique writers or images. HASY writer identities do not guarantee writer-disjoint evaluation; this is not its published 10-fold benchmark.
- Equals: 5,400 procedural training examples and 600 procedural validation examples. No real handwritten equals test data; this class is experimental.
- Total: **367,963 training presentations**, **30,304 validation examples**, **20,216 handwritten test examples**. Exact unique counts per class are recorded in the metrics.
- CNN: Conv48(3×3, ReLU), pool2, Conv96(3×3, ReLU), pool2, Dense192(ReLU), Dropout0.3, Dense83(softmax). Adam and batch size 256.
- Initial training: up to 24 epochs, early-stopped after 15 and restored the best validation-loss weights. Rotation, translation, zoom and random thinner/thicker ink were used.
- Fine-tuning: inverse training-class-frequency weights, lighter geometry and 15% thicker ink, Adam starting at 0.0001. Early-stopped after 9 epochs, restoring the best weights. No test examples were fitted.
- Simulated thicker-stroke tests improved all focused categories, but they are transformations of dataset glyphs, not independent real mouse drawings. See the comparison metrics.

The clean notebook `GlyphDex_v2_Colab.ipynb` has separate numbered setup, preparation, training, fine-tuning, evaluation, comparison and export cells. Optional validated website corrections can be imported before training; the current release used none.

## Runtime verification and artifacts

- `characters_cnn.keras` / `characters_cnn_weights.npz`: the new full model and NumPy weights.
- `baseline_cnn.keras` / `baseline_cnn_weights.npz`: the retained October 11 character model used in the mixed-mode blend. Its original training code remains preserved in the v1 notebook and script.
- `labels.json`: shared output ordering; never reorder it independently of weights.
- `verification.npz`: three held-out examples from each of the 82 handwritten classes, with candidate, baseline and blended Colab probabilities. These are parity fixtures, not the full test set.
- The NumPy runtime matched all 246 candidate and all 246 blended predicted labels. Maximum probability difference was **7.87e-6** for the new model and **2.03e-6** for the blend.
- Thirteen checks cover both exports, mixed routing, a previously confidently wrong canvas plus sign, the HTTP API, segmentation, uncertain outcomes, writer-separated exports and full-line rejection accounting.
- A 32-character mixed inference batch took 0.092 seconds locally; this is not a Render latency guarantee. The runtime processes four glyphs at a time to bound memory and needs no TensorFlow on Render.

## Interface and limitations

One line of separate, plain printed characters, up to 32 glyphs. Cursive, touching/overlapping letters, multiple lines, photos, arbitrary punctuation and mathematical expression layout remain unsupported. `!`, `?`, `@`, commas and parentheses are not trained classes. The app's `?` is an uncertainty placeholder.

Segmentation uses column gaps and can merge a small detached dot with a nearby taller stem. It does not reliably split touching characters. Always check the boxes; save a whole-line example when segmentation is wrong.

Conservative score/margin rules, known mixed-mode ambiguities and a filled-shape check flag some uncertain inputs. They are **not calibrated** and do not catch every confidently wrong or unsupported drawing. Copy requires resolving flagged glyphs. Alternative buttons and typed corrections are available. Model scores are estimates, not measured accuracy guarantees.

Checked character crops and explicitly labelled complete lines can be saved in browser local storage and exported. Nothing is automatically uploaded or used for training. `correction_data.py` validates labels and duplicates and requires real writer-separated groups for train/validation/test. `evaluate_lines.py` can evaluate only test writers through the full pipeline, counting spaces, rejected characters and request failures. Browser identifiers alone do not establish distinct real writers. No real website accuracy is claimed by this release.

## Sources and attribution

EMNIST: Gregory Cohen, Saeed Afshar, Jonathan Tapson and André van Schaik, [EMNIST: an extension of MNIST to handwritten letters](https://arxiv.org/abs/1702.05373), 2017. Original dataset: [NIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) / [Western Sydney University](https://rds.westernsydney.edu.au/Institutes/MARCS/BENS/EMNIST/). Colab rejected those download endpoints with HTTP 403; this run obtained the original IDX gzip format from the [Heliosoph EMNIST data mirror](https://huggingface.co/datasets/Heliosoph/EMNIST). Upstream NIST data originates from NIST Special Database 19. Mirror metadata lists CC BY 4.0; preserve attribution for redistributed verification samples.

HASYv2: Martin Thoma, [The HASYv2 dataset](https://arxiv.org/abs/1701.08380), 2017, [Zenodo release](https://doi.org/10.5281/zenodo.259444), MD5 `fddf23f36e24b5236f6b3a0880c778e3`. Derived from Detexify (Daniel Kirsch) and write-math recordings. The dataset is published under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/). HASY-derived verification glyphs retain that source attribution and database license; the full source database is available at the linked release. Original HASY glyphs were inverted, cropped, resized and centered here. Procedural equals examples were generated by this training script. Original DigiDex notebooks remain preserved separately.
