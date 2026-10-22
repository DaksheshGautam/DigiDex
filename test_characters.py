"""Meaningful checks for normalization, line segmentation and exported inference."""
import base64
import io
import json
import unittest
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from character_preprocessing import normalize_glyph, segment_line
from inference import DigitModel

ROOT = Path(__file__).parent

def payload(image, mode='mixed'):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return {'image':'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode(), 'mode':mode}

class SegmentationTests(unittest.TestCase):
    def test_dots_and_stacked_strokes_stay_with_character(self):
        image = Image.new('L',(240,100))
        draw = ImageDraw.Draw(image)
        draw.line((30,40,30,85),fill=255,width=5)
        draw.ellipse((27,20,33,26),fill=255)  # dotted i: two disconnected components
        draw.line((80,40,115,40),fill=255,width=5)
        draw.line((80,65,115,65),fill=255,width=5)  # equals: two stacked components
        draw.line((160,20,160,85),fill=255,width=5)
        glyphs, boxes = segment_line(image)
        self.assertEqual(len(boxes),3)
        self.assertEqual(glyphs.shape,(3,28,28,1))
        self.assertLess(boxes[0][1],30)
        self.assertGreater(boxes[0][3],80)
        self.assertLess(boxes[1][1],45)
        self.assertGreater(boxes[1][3],60)

    def test_blank_and_character_limit(self):
        with self.assertRaises(ValueError): segment_line(Image.new('L',(100,100)))
        image=Image.new('L',(400,100)); draw=ImageDraw.Draw(image)
        for i in range(33): draw.rectangle((i*12,20,i*12+3,60),fill=255)
        with self.assertRaises(ValueError): segment_line(image)

class CharacterModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference=np.load(ROOT/'models/characters/verification.npz',allow_pickle=False)
        cls.model=DigitModel(ROOT/'models/characters/characters_cnn_weights.npz')
        cls.labels=json.loads((ROOT/'models/characters/labels.json').read_text(encoding='utf-8'))

    def test_numpy_export_matches_colab_all_classes(self):
        result=np.concatenate([self.model.predict(self.reference['images'][i:i+4]) for i in range(0,len(self.reference['images']),4)])
        np.testing.assert_allclose(result,self.reference['probabilities'],atol=1e-5,rtol=1e-4)
        np.testing.assert_array_equal(result.argmax(1),self.reference['probabilities'].argmax(1))
        self.assertEqual(len(self.labels),83)

    def test_mixed_runtime_matches_exported_colab_routing(self):
        from app import character_probabilities
        if 'mixed_probabilities' not in self.reference:
            self.skipTest('Current export predates mixed-mode routing verification.')
        result=character_probabilities(self.reference['images'],'mixed')
        np.testing.assert_allclose(result,self.reference['mixed_probabilities'],atol=1e-5,rtol=1e-4)
        np.testing.assert_array_equal(result.argmax(1),self.reference['mixed_probabilities'].argmax(1))

    def test_web_reads_left_to_right_and_preserves_legacy(self):
        from app import app
        client=app.test_client()
        # Pick correctly classified real held-out glyphs across all three categories.
        ids=[]
        for character in ['A','7','+']:
            target=self.labels.index(character)
            candidates=np.flatnonzero((self.reference['labels']==target)&(self.reference['probabilities'].argmax(1)==target))
            self.assertGreater(len(candidates),0,character)
            ids.append(int(candidates[0]))
        line=Image.new('L',(360,100))
        for i,sample_id in enumerate(ids):
            glyph=Image.fromarray((self.reference['images'][sample_id,:,:,0]*255).astype('uint8')).resize((84,84))
            line.paste(glyph,(i*110+10,8))
        reply=client.post('/predict',json=payload(line))
        self.assertEqual(reply.status_code,200)
        self.assertEqual(''.join(c['suggestion'] for c in reply.json['characters']),'A7+')
        for c in reply.json['characters']:
            self.assertEqual(c['character']=='?',c['uncertain'])
        boxes=[c['box'] for c in reply.json['characters']]
        self.assertTrue(all(boxes[i][2]<boxes[i+1][0] for i in range(2)))
        self.assertEqual(client.get('/healthz').json['classes'],83)
        self.assertIn(b'Your handwriting.',client.get('/').data)
        self.assertEqual(client.post('/predict',json=payload(Image.new('L',(100,100)))).status_code,400)
        self.assertEqual(client.post('/predict',json=payload(line,'invalid')).status_code,400)
        self.assertEqual(client.post('/predict',json=[]).status_code,400)
        legacy=np.load(ROOT/'models/verification.npz',allow_pickle=False)
        image=Image.fromarray((legacy['images'][0,:,:,0]*255).astype('uint8')).resize((280,280))
        body=payload(image); body.pop('mode')
        self.assertEqual(client.post('/predict',json=body).json['digit'],int(legacy['labels'][0]))

if __name__=='__main__': unittest.main()
