import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from character_preprocessing import segment_line
from recognition_quality import assess
from correction_data import prepare_exports
from test_characters import payload


class QualityTests(unittest.TestCase):
    def test_shifted_dot_joins_stem_without_merging_next_letter(self):
        image=Image.new('L',(180,100));draw=ImageDraw.Draw(image)
        draw.rectangle((30,40,35,85),fill=255)
        draw.rectangle((40,22,45,27),fill=255)
        draw.line((85,20,85,85),fill=255,width=5)
        _,boxes=segment_line(image)
        self.assertEqual(len(boxes),2)
        self.assertEqual(boxes[0],[30,22,46,86])

    def test_weak_wrong_mode_and_ambiguous_predictions_are_rejected(self):
        labels=list('0OAB');glyph=np.zeros((28,28,1));glyph[4:24,12:14]=1
        self.assertTrue(assess(np.array([.1,.1,.45,.35]),[2,3,0],labels,'uppercase',glyph))
        self.assertTrue(assess(np.array([.98,.01,.005,.005]),[0,1,2],labels,'mixed',glyph))
        self.assertFalse(assess(np.array([.98,.01,.005,.005]),[0,1,2],labels,'digits',glyph))
        self.assertFalse(assess(np.array([.005,.005,.98,.01]),[2,3,0],labels,'mixed',glyph))

    def test_exports_keep_writer_groups_separate_and_reject_conflicting_duplicates(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            examples=[]
            for i in range(3):
                image=Image.new('L',(64,64));draw=ImageDraw.Draw(image)
                draw.line((8,10,30+i*8,50),fill=255,width=3+i)
                examples.append({'label':'A','writer_group':f'writer{i}','image':payload(image)['image']})
            path=Path(folder)/'examples.json'
            path.write_text(json.dumps({'schema_version':1,'examples':examples}),encoding='utf-8')
            arrays,summary=prepare_exports([path],['A'])
            self.assertEqual({s['writers'] for s in [summary[k] for k in ['train','validation','test']]},{1})
            self.assertEqual(sum(len(arrays[k+'_images']) for k in ['train','validation','test']),3)
            examples.append(dict(examples[0],writer_group='another-writer'))
            path.write_text(json.dumps({'schema_version':1,'examples':examples}),encoding='utf-8')
            with self.assertRaises(ValueError): prepare_exports([path],['A'])

    def test_unsupported_corrections_fail_validation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path=Path(folder)/'bad.json'
            path.write_text(json.dumps({'schema_version':1,'examples':[{'label':'@','writer_group':'one'}]}))
            with self.assertRaises(ValueError): prepare_exports([path],['A'])
