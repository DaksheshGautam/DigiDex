import base64
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from flask import Flask
from PIL import Image,ImageDraw
from dataset_routes import install_dataset,attempts
from dataset_store import SQLiteTestStore,validate_sample,DatasetValidationError
from correction_data import prepare_exports


def sample(label='+',offset=0,kind='glyph'):
    image=Image.new('L',(100,100));draw=ImageDraw.Draw(image)
    draw.line((15,50,85,50),fill=255,width=7+offset)
    draw.line((50,15,50,85),fill=255,width=7+offset)
    data=io.BytesIO();image.save(data,format='PNG')
    return dict(kind=kind,writer_group='owner',drawing_id='drawing-'+str(offset),mode='symbols',
                label=label,text=label,image='data:image/png;base64,'+base64.b64encode(data.getvalue()).decode(),
                strokes=[{'width':7,'points':[[15,50],[85,50]]}])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.addCleanup(self.temp.cleanup)
        self.store=SQLiteTestStore(Path(self.temp.name)/'test.sqlite');self.store.initialize()
        self.key='private-test-key-with-enough-length'
        self.app=Flask(__name__);install_dataset(self.app,list('+-ABC'))
        self.app.config.update(TESTING=True,SECRET_KEY='session-secret-for-tests',
                               DATASET_KEY_HASH=hashlib.sha256(self.key.encode()).hexdigest())
        self.app.extensions['dataset_store']=self.store
        self.client=self.app.test_client();attempts.clear()
        self.base={'base_url':'https://example.test'}

    def login(self):
        response=self.client.post('/dataset/session',json={'key':self.key},**self.base)
        self.assertEqual(response.status_code,200)
        return response.json['csrf']

    def test_private_access_csrf_and_cross_site_requests(self):
        self.assertEqual(self.client.get('/dataset/export',**self.base).status_code,401)
        self.assertEqual(self.client.post('/dataset/session',json={'key':self.key},headers={'Origin':'https://evil.test'},**self.base).status_code,403)
        csrf=self.login()
        cookies=self.client.get('/dataset/status',**self.base)
        self.assertTrue(cookies.json['authenticated'])
        self.assertEqual(self.client.post('/dataset/samples',json={'samples':[sample()]},**self.base).status_code,401)
        response=self.client.post('/dataset/samples',json={'samples':[sample()]},headers={'X-Dataset-CSRF':csrf},**self.base)
        self.assertEqual(response.json['characters'],1)
        exported=self.client.get('/dataset/export',**self.base).json
        self.assertEqual(exported['examples'][0]['strokes'][0]['width'],7)
        self.assertNotIn(self.key,json.dumps(exported))

    def test_relabel_upserts_and_export_paginates_without_duplicates(self):
        headers={'Authorization':'Bearer '+self.key}
        for offset in range(5):
            self.assertEqual(self.client.post('/dataset/samples',headers=headers,json={'samples':[sample(offset=offset)]},**self.base).status_code,200)
        self.client.post('/dataset/samples',headers=headers,json={'samples':[sample(label='-')]},**self.base)
        self.assertEqual(self.store.counts()['characters'],5)
        cursor='';records=[]
        while True:
            page=self.client.get('/dataset/export',query_string={'limit':2,'cursor':cursor},headers=headers,**self.base).json
            records.extend(page['examples']);cursor=page['next_cursor']
            if not cursor:break
        self.assertEqual(len({r['id'] for r in records}),5)
        self.assertEqual(sum(r['label']=='-' for r in records),1)

    def test_invalid_batch_is_atomic_and_failure_does_not_discard_owner_session(self):
        csrf=self.login();bad=sample();bad['label']='unsupported'
        self.assertEqual(self.client.post('/dataset/samples',json={'samples':[sample(),bad]},headers={'X-Dataset-CSRF':csrf},**self.base).status_code,400)
        self.assertEqual(self.store.counts()['characters'],0)
        def unavailable():raise RuntimeError('secret database URI')
        self.store.counts=unavailable
        response=self.client.get('/dataset/status',**self.base)
        self.assertTrue(response.json['authenticated']);self.assertEqual(response.json['csrf'],csrf)
        self.assertNotIn('secret database URI',response.text)

    def test_empty_image_and_nan_strokes_are_rejected(self):
        bad=sample();bad['strokes'][0]['points']=[[float('nan'),3]]
        with self.assertRaises(DatasetValidationError):validate_sample(bad,list('+-'))
        image=Image.new('L',(20,20));data=io.BytesIO();image.save(data,format='PNG')
        bad=sample();bad['image']='data:image/png;base64,'+base64.b64encode(data.getvalue()).decode()
        with self.assertRaises(DatasetValidationError):validate_sample(bad,list('+-'))

    def test_personalization_keeps_whole_drawings_together(self):
        records=[]
        for offset in range(6):
            _,record=validate_sample(sample(offset=offset),list('+-'))
            records.append(record)
        path=Path(self.temp.name)/'private.json';path.write_text(json.dumps({'schema_version':1,'examples':records}))
        with self.assertRaises(ValueError):prepare_exports([path],list('+-'))
        arrays,report=prepare_exports([path],list('+-'),'drawing')
        self.assertEqual(sum(len(arrays[s+'_labels']) for s in ('train','validation','test')),6)
        self.assertEqual(report['split_strategy'],'drawing')
        self.assertIn('does not measure accuracy on unseen writers',report['note'])


if __name__=='__main__':unittest.main()
