"""Run in Colab only. EMNIST ByClass + selected HASY symbols; no local training."""
import base64
import csv
import gzip
import hashlib
import io
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageOps
from IPython.display import HTML, display

tf.keras.utils.set_random_seed(42)
rng = np.random.default_rng(42)
assert tf.config.list_physical_devices('GPU'), 'Select Runtime > Change runtime type > T4 GPU, then run again.'
print('Training in Google Colab on:', tf.config.list_physical_devices('GPU'), flush=True)
labels = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
symbol_map = {'+': '+', '-': '-', '/': '/', '<': '<', '>': '>', r'\#': '#', r'\$': '$',
              r'\%': '%', r'\&': '&', '[': '[', ']': ']', r'\{': '{', r'\}': '}',
              r'\times': '×', r'\div': '÷', r'\pm': '±', r'\leq': '≤', r'\geq': '≥',
              r'\neq': '≠', r'\approx': '≈'}
labels += list(symbol_map.values()) + ['=']

def download(url, filename):
    if not Path(filename).exists():
        print('Downloading', filename, flush=True)
        # Publish cached downloads only after the entire transfer succeeds.
        urllib.request.urlretrieve(url, filename + '.part')
        Path(filename + '.part').replace(filename)

if not Path('emnist.zip').exists():
    # The NIST and university endpoints currently reject Colab requests (HTTP 403).
    # This data-only mirror preserves the original IDX gzip files.
    with zipfile.ZipFile('emnist.zip.part', 'w', zipfile.ZIP_STORED) as emnist_archive:
        for split in ['train', 'test']:
            for kind in ['images', 'labels']:
                filename = f'emnist-byclass-{split}-{kind}-idx{3 if kind == "images" else 1}-ubyte.gz'
                download('https://huggingface.co/datasets/Heliosoph/EMNIST/resolve/main/' + filename, filename)
                emnist_archive.write(filename, filename)
    Path('emnist.zip.part').replace('emnist.zip')
download('https://zenodo.org/records/259444/files/HASYv2.tar.bz2?download=1', 'HASYv2.tar.bz2')
assert hashlib.md5(Path('HASYv2.tar.bz2').read_bytes()).hexdigest() == 'fddf23f36e24b5236f6b3a0880c778e3'
parts = {k: [[], []] for k in ['train', 'validation', 'test']}
counts = {}

def add(split, images, class_id):
    parts[split][0].extend(images)
    parts[split][1].extend([class_id] * len(images))

with zipfile.ZipFile('emnist.zip') as archive:
    def idx(split, kind):
        filename = f'emnist-byclass-{split}-{kind}-idx{3 if kind == "images" else 1}-ubyte.gz'
        name = next(n for n in archive.namelist() if n.endswith(filename))
        raw = gzip.decompress(archive.read(name))
        array = np.frombuffer(raw, np.uint8, offset=16 if kind == 'images' else 8)
        return array.reshape(-1, 28, 28) if kind == 'images' else array
    # Deterministic, capped random class sampling; validation comes only from official train.
    for split in ['train', 'test']:
        images, targets = idx(split, 'images'), idx(split, 'labels')
        for class_id in range(62):
            candidates = np.flatnonzero(targets == class_id)
            selected = rng.permutation(candidates)[:min(len(candidates), 6000 if split == 'train' else 300)]
            prepared = [normalize_glyph(Image.fromarray(images[i].T)) for i in selected]
            if split == 'train':
                nval = max(1, len(prepared) // 10)
                add('validation', prepared[:nval], class_id)
                add('train', prepared[nval:], class_id)
                counts[labels[class_id]] = {'train_unique': len(prepared)-nval, 'validation': nval}
            else:
                add('test', prepared, class_id)
                counts[labels[class_id]]['test'] = len(prepared)
        print('EMNIST', split, 'normalized.', flush=True)
        del images, targets

with tarfile.open('HASYv2.tar.bz2') as archive:
    # Index once; images remain in archive and only the selected glyphs are read.
    members = {m.name: m for m in archive.getmembers()}
    def rows(split):
        return list(csv.DictReader(io.TextIOWrapper(archive.extractfile(members[f'classification-task/fold-1/{split}.csv']))))
    train_rows, test_rows = rows('train'), rows('test')
    wanted = {'hasy-data/' + Path(r['path']).name for r in train_rows + test_rows if r['latex'] in symbol_map}
    image_bytes = {}
    # One sequential pass avoids repeatedly decompressing the archive for random reads.
    with tarfile.open('HASYv2.tar.bz2', mode='r|bz2') as stream:
        for member in stream:
            if member.name in wanted:
                image_bytes[member.name] = stream.extractfile(member).read()
    for latex, character in symbol_map.items():
        class_id = labels.index(character)
        def glyphs(data):
            selected = [r for r in data if r['latex'] == latex]
            return [normalize_glyph(ImageOps.invert(Image.open(io.BytesIO(image_bytes['hasy-data/' + Path(r['path']).name])).convert('L'))) for r in selected]
        train_images, test_images = glyphs(train_rows), glyphs(test_rows)
        assert len(train_images) > 30 and len(test_images) > 0, character
        order = rng.permutation(len(train_images))
        nval = max(5, len(train_images) // 10)
        validation = [train_images[i] for i in order[:nval]]
        training = [train_images[i] for i in order[nval:]]
        add('validation', validation, class_id)
        add('test', test_images, class_id)
        # Repeated training examples balance scarce symbols; augmentation adds variation.
        add('train', [training[i] for i in rng.choice(len(training), 5400, replace=True)], class_id)
        counts[character] = {'train_unique': len(training), 'train_presentations': 5400, 'validation': len(validation), 'test': len(test_images)}
    print('HASY fold-1 symbols normalized.', flush=True)

# HASY has no equals class. This simple two-stroke class uses procedural examples.
# It has no human handwriting benchmark and is explicitly experimental in the app.
def equals_glyph():
    image = Image.new('L', (64, 64))
    draw = ImageDraw.Draw(image)
    width = int(rng.integers(2, 8))
    x1, x2 = int(rng.integers(4, 16)), int(rng.integers(47, 61))
    y1, y2 = int(rng.integers(15, 27)), int(rng.integers(35, 49))
    slope = int(rng.integers(-5, 6))
    for y in [y1, y2]:
        draw.line([(x1 + int(rng.integers(-3, 4)), y), ((x1+x2)//2, y+slope//2+int(rng.integers(-2,3))), (x2, y+slope)], fill=255, width=width)
    return normalize_glyph(image)
add('train', [equals_glyph() for _ in range(5400)], labels.index('='))
add('validation', [equals_glyph() for _ in range(600)], labels.index('='))
counts['='] = {'train_synthetic': 5400, 'validation_synthetic': 600, 'test': 0}

x_train, y_train = np.stack(parts['train'][0]), np.array(parts['train'][1])
x_val, y_val = np.stack(parts['validation'][0]), np.array(parts['validation'][1])
x_test, y_test = np.stack(parts['test'][0]), np.array(parts['test'][1])
del parts
website_test = None
if Path('website_corrections.npz').exists():
    # Upload only an export prepared with correction_data.py and verified writer groups.
    with np.load('website_corrections.npz',allow_pickle=False) as web:
        x_train = np.concatenate([x_train, web['train_images']])
        y_train = np.concatenate([y_train, web['train_labels']])
        x_val = np.concatenate([x_val, web['validation_images']])
        y_val = np.concatenate([y_val, web['validation_labels']])
        website_test = (web['test_images'].copy(),web['test_labels'].copy())
    print('Included writer-separated website corrections. Website test glyphs remain held out.',flush=True)
print('DATA_READY:', len(x_train), len(x_val), len(x_test), 'classes:', len(labels), flush=True)
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(42)
augmentation = tf.keras.Sequential([tf.keras.layers.RandomRotation(.04, fill_mode='constant'), tf.keras.layers.RandomTranslation(.08, .08, fill_mode='constant'), tf.keras.layers.RandomZoom(.12, fill_mode='constant')])
def augment_batch(x, y):
    x = augmentation(x, training=True)
    # Vary ink thickness without changing labels; no reflections or large rotations.
    thick = tf.nn.max_pool2d(x, 2, 1, 'SAME')
    thin = -tf.nn.max_pool2d(-x, 2, 1, 'SAME')
    pick = tf.random.uniform([tf.shape(x)[0],1,1,1])
    return tf.where(pick < .15, thin, tf.where(pick > .75, thick, x)), y

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train), seed=42).batch(256).map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
model = tf.keras.Sequential([
    tf.keras.layers.Input((28,28,1)), tf.keras.layers.Conv2D(48,3,activation='relu'),
    tf.keras.layers.MaxPooling2D(2), tf.keras.layers.Conv2D(96,3,activation='relu'),
    tf.keras.layers.MaxPooling2D(2), tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(192,activation='relu'), tf.keras.layers.Dropout(.3),
    tf.keras.layers.Dense(len(labels),activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, validation_data=(x_val,y_val), epochs=24, verbose=2,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True),
                               tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=2,factor=.5,min_lr=1e-5)])
# Balance classes before fine-tuning: the larger capped sample is not uniform.
class_counts = np.bincount(y_train,minlength=len(labels))
class_weights = {i:float(len(y_train)/(len(labels)*n)) for i,n in enumerate(class_counts)}
augmentation = tf.keras.Sequential([tf.keras.layers.RandomRotation(.025,fill_mode='constant'),
    tf.keras.layers.RandomTranslation(.04,.04,fill_mode='constant'),tf.keras.layers.RandomZoom(.08,fill_mode='constant')])
def balanced_augment(x,y):
    x=augmentation(x,training=True)
    pick=tf.random.uniform([tf.shape(x)[0],1,1,1])
    thick=tf.nn.max_pool2d(x,2,1,'SAME')
    return tf.where(pick>.85,thick,x),y
balanced_ds=tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(len(x_train),seed=43).batch(256).map(balanced_augment,num_parallel_calls=2).prefetch(2)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
first_history={k:list(v) for k,v in history.history.items()}
balanced_history=model.fit(balanced_ds,validation_data=(x_val,y_val),class_weight=class_weights,epochs=10,verbose=2,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=1,factor=.5,min_lr=1e-5)])
for key in history.history:
    history.history[key] += balanced_history.history.get(key,[])
print('Balanced fine-tuning complete.',flush=True)

probabilities = model.predict(x_test,batch_size=256,verbose=0)
correct = probabilities.argmax(1) == y_test
categories = {}
for name, mask in [('digits',y_test<10),('letters',(y_test>=10)&(y_test<62)),('symbols',y_test>=62)]:
    categories[name] = {'examples':int(mask.sum()),'accuracy':float(correct[mask].mean())}
mode_accuracy = {}
for name, low, high in [('digits',0,10),('uppercase',10,36),('lowercase',36,62),('symbols',62,83)]:
    mask = (y_test>=low)&(y_test<high)
    mode_accuracy[name] = {'examples':int(mask.sum()), 'accuracy':float((probabilities[mask,low:high].argmax(1)+low==y_test[mask]).mean())}
metrics = {'dataset':'EMNIST ByClass capped per-class sample + HASYv2 fold-1 subset', 'seed':42,
           'training_environment':'Google Colab', 'tensorflow_version':tf.__version__,
           'train_presentations':len(x_train), 'validation_examples':len(x_val), 'test_examples':len(x_test),
           'epochs':len(history.history['loss']), 'batch_size':256,'categories':categories,'mode_accuracy':mode_accuracy,
           'counts':counts, 'history':history.history, 'experimental_symbols':['='],
           'per_class_accuracy':{labels[i]:float(correct[y_test==i].mean()) for i in range(82)},
           'test_note':'Held-out individual characters, not whole-line accuracy. Equals has synthetic training/validation only and is excluded from test metrics. HASY fold-1 only; not the 10-fold benchmark.'}
display(HTML('<h3>Held-out accuracy by recognition mode</h3><table><tr><th>Mode</th><th>Examples</th><th>Accuracy</th></tr>' + ''.join(f"<tr><td>{name}</td><td>{v['examples']:,}</td><td>{v['accuracy']*100:.2f}%</td></tr>" for name,v in mode_accuracy.items()) + '</table><p>Correct focused mode assumed. Individual glyphs, not complete-line or real website accuracy.</p>'))
import matplotlib.pyplot as plt
plt.figure(figsize=(8,3))
plt.plot(history.history['val_loss'],label='Validation loss')
plt.axvline(len(first_history['loss'])-.5,color='gray',linestyle='--',label='Balanced fine-tuning starts')
plt.xlabel('Epoch (zero-based)'); plt.ylabel('Loss'); plt.legend(); plt.grid(alpha=.2); plt.tight_layout(); plt.show()

baseline_url = 'https://raw.githubusercontent.com/DaksheshGautam/GlyphDex/5d9871689a129986c97478151998074e2354f0ec/models/characters/characters_cnn.keras'
download(baseline_url, 'baseline.keras')
baseline = tf.keras.models.load_model('baseline.keras')
old_probabilities = baseline.predict(x_test, batch_size=256, verbose=0)
comparison = {'note':'Same held-out samples for both models; transformed samples are simulated stroke changes, not real website drawings.', 'modes':{}}
comparison['mixed_by_category'] = {}
for name, low, high in [('digits',0,10),('uppercase',10,36),('lowercase',36,62),('symbols',62,83)]:
    mask = (y_test>=low)&(y_test<high)
    comparison['mixed_by_category'][name] = {'examples':int(mask.sum()),
        'baseline':float((old_probabilities[mask].argmax(1)==y_test[mask]).mean()),
        'candidate':float((probabilities[mask].argmax(1)==y_test[mask]).mean())}
for name, low, high in [('digits',0,10),('uppercase',10,36),('lowercase',36,62),('symbols',62,83)]:
    mask = (y_test>=low)&(y_test<high)
    comparison['modes'][name] = {
        'examples':int(mask.sum()),
        'baseline':float((old_probabilities[mask,low:high].argmax(1)+low==y_test[mask]).mean()),
        'candidate':mode_accuracy[name]['accuracy']}
variant_ids = np.arange(0,len(x_test),4)
variant_x = tf.nn.max_pool2d(x_test[variant_ids], 2, 1, 'SAME').numpy()
variant_y = y_test[variant_ids]
variant_old = baseline.predict(variant_x,batch_size=256,verbose=0)
variant_new = model.predict(variant_x,batch_size=256,verbose=0)
comparison['thicker_strokes'] = {}
for name, low, high in [('digits',0,10),('uppercase',10,36),('lowercase',36,62),('symbols',62,83)]:
    mask = (variant_y>=low)&(variant_y<high)
    comparison['thicker_strokes'][name] = {'examples':int(mask.sum()),
        'baseline':float((variant_old[mask,low:high].argmax(1)+low==variant_y[mask]).mean()),
        'candidate':float((variant_new[mask,low:high].argmax(1)+low==variant_y[mask]).mean())}
# Select a mixed-mode blend using validation glyphs held out from BOTH runs.
# The old validation prefix remains in the larger new validation prefix for EMNIST.
# HASY validation samples changed with resampling RNG, so exclude them here.
download('https://raw.githubusercontent.com/DaksheshGautam/GlyphDex/5d9871689a129986c97478151998074e2354f0ec/models/characters/metrics.json','baseline_metrics.json')
baseline_metrics=json.loads(Path('baseline_metrics.json').read_text())
shared_ids=np.concatenate([np.flatnonzero(y_val==i)[:baseline_metrics['counts'][labels[i]]['validation']] for i in range(62)])
shared_y=y_val[shared_ids]
shared_old=baseline.predict(x_val[shared_ids],batch_size=256,verbose=0)
shared_new=model.predict(x_val[shared_ids],batch_size=256,verbose=0)
blend_trials=[]
for weight in [0.,.25,.5,.75,1.]:
    blend=shared_new*weight+shared_old*(1-weight)
    predicted=blend.argmax(1)
    per_class=[float((predicted[shared_y==i]==i).mean()) for i in range(62)]
    groups={name:float((predicted[(shared_y>=low)&(shared_y<high)]==shared_y[(shared_y>=low)&(shared_y<high)]).mean()) for name,low,high in [('digits',0,10),('uppercase',10,36),('lowercase',36,62)]}
    blend_trials.append({'candidate_weight':weight,'macro_validation_accuracy':float(np.mean(per_class)),'groups':groups})
baseline_groups=blend_trials[0]['groups']
eligible=[t for t in blend_trials if all(t['groups'][k]>=baseline_groups[k]-.005 for k in baseline_groups)]
chosen=max(eligible,key=lambda t:t['macro_validation_accuracy'])
mixed_weight=chosen['candidate_weight']
mixed_probabilities=probabilities*mixed_weight+old_probabilities*(1-mixed_weight)
mixed_groups={}
for name,low,high in [('digits',0,10),('uppercase',10,36),('lowercase',36,62),('symbols',62,83)]:
    mask=(y_test>=low)&(y_test<high)
    mixed_groups[name]={'examples':int(mask.sum()),'baseline':comparison['mixed_by_category'][name]['baseline'],
        'candidate':float((mixed_probabilities[mask].argmax(1)==y_test[mask]).mean())}
if any(v['candidate']<v['baseline']-.005 for v in mixed_groups.values()):
    mixed_weight=0.
    mixed_probabilities=old_probabilities.copy()
    for v in mixed_groups.values():v['candidate']=v['baseline']
comparison['mixed_runtime']=mixed_groups
comparison['mixed_routing']={'candidate_weight':mixed_weight,'validation_choice':chosen,'validation_trials':blend_trials,
    'note':'Blend selected on shared held-out EMNIST validation. Regression gate falls back to baseline if a test category drops more than 0.5 points. Focused modes use the new model.'}
for title,key in [('Focused recognition modes','modes'),('New model alone in mixed mode','mixed_by_category'),('Selected mixed-mode runtime','mixed_runtime'),('Simulated thicker strokes, focused modes','thicker_strokes')]:
    display(HTML('<h3>'+title+'</h3><table><tr><th>Group</th><th>Baseline</th><th>Candidate</th><th>Change (points)</th></tr>' + ''.join(f"<tr><td>{name}</td><td>{v['baseline']*100:.2f}%</td><td>{v['candidate']*100:.2f}%</td><td>{(v['candidate']-v['baseline'])*100:+.2f}</td></tr>" for name,v in comparison[key].items()) + '</table>'))
print('Baseline comparison complete. Real website accuracy requires independent writer samples.',flush=True)
metrics['comparison'] = comparison
metrics['routing']={'focused':'v2','mixed_candidate_weight':mixed_weight}
metrics['version'] = 'v2'
metrics['architecture'] = 'Conv48, pool, Conv96, pool, Dense192, Dropout0.3, Dense83'
metrics['augmentation'] = 'rotation, translation, zoom, random ink dilation/erosion'
metrics['fine_tuning'] = {'epochs':len(balanced_history.history['loss']),'class_weights':'inverse training-class frequency','augmentation':'lighter geometry, 15% thicker ink','initial_epochs':len(first_history['loss'])}
metrics['real_website_test_examples'] = 0
website_test = globals().get('website_test')
if website_test is not None:
    website_x,website_y = website_test
    website_probabilities = model.predict(website_x,batch_size=256,verbose=0)
    metrics['real_website_test_examples'] = len(website_y)
    metrics['website_modes'] = {}
    for name,low,high in [('digits',0,10),('uppercase',10,36),('lowercase',36,62),('symbols',62,83)]:
        mask = (website_y>=low)&(website_y<high)
        if mask.any():
            metrics['website_modes'][name] = {'examples':int(mask.sum()),
                'accuracy':float((website_probabilities[mask,low:high].argmax(1)+low==website_y[mask]).mean())}
out = Path('glyphdex_characters_v2'); out.mkdir(exist_ok=True)
model.save(out/'characters_cnn.keras')
baseline.save(out/'baseline_cnn.keras')
np.savez_compressed(out/'baseline_cnn_weights.npz',**{f'w{i}':w for i,w in enumerate(baseline.get_weights())})
np.savez_compressed(out/'characters_cnn_weights.npz',**{f'w{i}':w for i,w in enumerate(model.get_weights())})
(out/'labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2))
(out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
sample_ids = np.concatenate([np.flatnonzero(y_test==i)[:3] for i in range(82)])
np.savez_compressed(out/'verification.npz',images=x_test[sample_ids],labels=y_test[sample_ids],probabilities=probabilities[sample_ids],baseline_probabilities=old_probabilities[sample_ids],mixed_probabilities=mixed_probabilities[sample_ids])
with zipfile.ZipFile('glyphdex_characters_v2.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for file in out.iterdir(): archive.write(file,file.name)
print('MODEL_READY: glyphdex_characters_v2.zip',flush=True)
blob = base64.b64encode(Path('glyphdex_characters_v2.zip').read_bytes()).decode()
display(HTML('<a id="character-model-export" download="glyphdex_characters_v2.zip" href="data:application/zip;base64,'+blob+'">Download trained character model</a>'))
