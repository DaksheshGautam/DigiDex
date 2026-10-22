"""Private correction storage. Production requires PostgreSQL with verified TLS."""
import base64
import hashlib
import io
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from character_preprocessing import normalize_glyph

MODES={'mixed','digits','uppercase','lowercase','symbols'}
IDENTIFIER=re.compile(r'^[A-Za-z0-9_.:-]{1,100}$')


class DatasetValidationError(ValueError):
    pass


def validate_sample(sample, labels):
    if not isinstance(sample,dict):
        raise DatasetValidationError('Invalid correction.')
    kind=sample.get('kind')
    writer=sample.get('writer_group','')
    drawing=sample.get('drawing_id','')
    if kind not in ('glyph','line') or not isinstance(writer,str) or not IDENTIFIER.fullmatch(writer) or not isinstance(drawing,str) or not IDENTIFIER.fullmatch(drawing):
        raise DatasetValidationError('A correction needs a drawing ID and writer group.')
    mode=sample.get('mode')
    label=sample.get('label') if kind=='glyph' else sample.get('text')
    if not isinstance(label,str) or not label or label.strip()!=label or mode not in MODES:
        raise DatasetValidationError('Enter the corrected character or full line.')
    if kind=='glyph' and label not in labels:
        raise DatasetValidationError('Use one supported character per box.')
    if kind=='line' and (len(label)>64 or len(label.replace(' ',''))>32 or any(c not in labels and c!=' ' for c in label)):
        raise DatasetValidationError('Use up to 32 supported characters for a line.')
    uri=sample.get('image','')
    if not isinstance(uri,str) or not uri.startswith('data:image/png;base64,') or len(uri)>360000:
        raise DatasetValidationError('The drawing must be a small PNG.')
    try:
        raw=base64.b64decode(uri.split(',',1)[1],validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.format!='PNG' or max(image.size)>1024 or min(image.size)<1:
                raise DatasetValidationError('Drawing dimensions are invalid.')
            image=image.convert('L')
            normalized=normalize_glyph(image)
            width,height=image.size
            output=io.BytesIO();image.save(output,format='PNG')
            canonical=(normalized*255).round().astype('uint8').tobytes() if kind=='glyph' else f'{width}:{height}:'.encode()+image.tobytes()
    except (ValueError,OSError,UnidentifiedImageError,Image.DecompressionBombError) as exc:
        raise DatasetValidationError('Draw a clear character before correcting it.') from exc
    # Updating a label for the same drawing replaces the previous label.
    # The label and selected mode intentionally do not enter the deduplication key.
    key=hashlib.sha256(writer.encode()+b'|'+kind.encode()+b'|'+canonical).hexdigest()
    item={'id':key,'kind':kind,'writer_group':writer,'drawing_id':drawing,'mode':mode,
          'image':'data:image/png;base64,'+base64.b64encode(output.getvalue()).decode(),
          'width':width,'height':height,'source':'explicit_correction'}
    item['label' if kind=='glyph' else 'text']=label
    suggestion=sample.get('model_suggestion')
    if suggestion in labels:item['model_suggestion']=suggestion
    box=sample.get('box')
    if isinstance(box,list) and len(box)==4 and all(isinstance(v,int) and 0<=v<=1024 for v in box) and box[0]<box[2] and box[1]<box[3]:
        item['box']=box
    stroke_data=sample.get('strokes',[])
    if not isinstance(stroke_data,list) or len(stroke_data)>512:
        raise DatasetValidationError('Too many drawing strokes.')
    points_total=0;clean_strokes=[]
    for stroke in stroke_data:
        if not isinstance(stroke,dict):raise DatasetValidationError('Invalid stroke.')
        points=stroke.get('points',[]);pen=stroke.get('width')
        if not isinstance(pen,(int,float)) or not 0<pen<=40 or not isinstance(points,list) or len(points)>5000:
            raise DatasetValidationError('Invalid stroke data.')
        points_total+=len(points)
        if points_total>20000:raise DatasetValidationError('Too many stroke points.')
        if any(not isinstance(p,list) or len(p)!=2 or any(not isinstance(v,(int,float)) or not -1024<=v<=2048 for v in p) for p in points):
            raise DatasetValidationError('Invalid stroke coordinates.')
        clean_strokes.append({'width':pen,'points':points})
    item['strokes']=clean_strokes
    return key,item


class DatasetStore:
    def __init__(self,url,ca_pem):
        if not url.startswith(('postgres://','postgresql://')) or not ca_pem.strip():
            raise ValueError('PostgreSQL URL and certificate are required.')
        self.url=url
        with tempfile.NamedTemporaryFile(mode='w',suffix='.pem',delete=False,encoding='utf-8') as cert:
            cert.write(ca_pem);self.ca_path=cert.name
        os.chmod(self.ca_path,0o600)

    @contextmanager
    def connection(self):
        import psycopg
        with psycopg.connect(self.url,sslmode='verify-full',sslrootcert=self.ca_path,connect_timeout=8,
                             options='-c statement_timeout=10000') as connection:
            yield connection

    def execute(self,connection,sql,values=()):
        return connection.execute(sql,values)

    def initialize(self):
        """Run once with the dedicated database owner, not the web-service role."""
        with self.connection() as connection:
            self.execute(connection,'''CREATE TABLE IF NOT EXISTS glyphdex_samples (
                sample_key VARCHAR(64) PRIMARY KEY,
                kind VARCHAR(8) NOT NULL CHECK (kind IN ('glyph','line')),
                writer_group VARCHAR(100) NOT NULL,
                payload TEXT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP)''')
            self.execute(connection,'CREATE INDEX IF NOT EXISTS glyphdex_samples_kind ON glyphdex_samples(kind)')

    def put(self,items):
        with self.connection() as connection:
            for key,item in items:
                if not self.execute(connection,'SELECT 1 FROM glyphdex_samples WHERE sample_key=%s',(key,)).fetchone():
                    count=self.execute(connection,'SELECT count(*) FROM glyphdex_samples').fetchone()[0]
                    if count>=50000:raise DatasetValidationError('The private dataset limit has been reached. Export it before adding more.')
                self.execute(connection,'''INSERT INTO glyphdex_samples(sample_key,kind,writer_group,payload)
                    VALUES (%s,%s,%s,%s) ON CONFLICT(sample_key) DO UPDATE SET
                    payload=excluded.payload,updated_at=CURRENT_TIMESTAMP''',
                    (key,item['kind'],item['writer_group'],json.dumps(item,ensure_ascii=False,allow_nan=False)))
        return self.counts()

    def counts(self):
        with self.connection() as connection:
            rows=self.execute(connection,'SELECT kind,count(*) FROM glyphdex_samples GROUP BY kind').fetchall()
        counts=dict(rows)
        return {'characters':counts.get('glyph',0),'lines':counts.get('line',0)}

    def export_page(self,cursor='',limit=200):
        if cursor and not re.fullmatch('[0-9a-f]{64}',cursor):raise DatasetValidationError('Invalid export cursor.')
        with self.connection() as connection:
            rows=self.execute(connection,'''SELECT sample_key,payload FROM glyphdex_samples
                WHERE sample_key>%s ORDER BY sample_key LIMIT %s''',(cursor,limit+1)).fetchall()
        page=rows[:limit]
        result={'schema_version':1,'examples':[],'lines':[],
                'next_cursor':page[-1][0] if len(rows)>limit else None}
        for key,payload in page:
            item=json.loads(payload)
            result['examples' if item['kind']=='glyph' else 'lines'].append(item)
        return result


class SQLiteTestStore(DatasetStore):
    """Used exclusively by local tests; production has no ephemeral fallback."""
    def __init__(self,path):self.path=Path(path)

    @contextmanager
    def connection(self):
        import sqlite3
        connection=sqlite3.connect(self.path)
        try:
            with connection:yield connection
        finally:connection.close()

    def execute(self,connection,sql,values=()):
        return connection.execute(sql.replace('%s','?'),values)
