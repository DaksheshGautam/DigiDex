"""Validate opt-in browser exports and split by writer, never by glyph.

This prepares data only; model fitting belongs in Colab.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from character_preprocessing import normalize_glyph


def prepare_exports(paths, labels):
    records, seen, writer_splits = [], {}, {}
    for path in paths:
        payload=json.loads(Path(path).read_text(encoding='utf-8'))
        if payload.get('schema_version') != 1 or not isinstance(payload.get('examples'),list):
            raise ValueError('Unsupported corrections file format.')
        for item in payload['examples']:
            label=item.get('label')
            writer=item.get('writer_group')
            if label not in labels or not isinstance(writer,str) or not writer or len(writer)>128:
                raise ValueError('Every example needs a supported label and writer group.')
            uri=item.get('image','')
            if not isinstance(uri,str) or len(uri)>500000 or not uri.startswith('data:image/png;base64,'):
                raise ValueError('Expected a small PNG glyph.')
            raw=base64.b64decode(uri.split(',',1)[1],validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                if max(image.size)>1024:
                    raise ValueError('Glyph is too large.')
                glyph=normalize_glyph(image)
            digest=hashlib.sha256(glyph.tobytes()).hexdigest()
            if digest in seen:
                if seen[digest] != (writer,label):
                    raise ValueError('Duplicate drawing has conflicting labels or writer groups; resolve before splitting.')
                continue
            seen[digest]=(writer,label)
            records.append((writer,labels.index(label),glyph))
    writers=sorted({r[0] for r in records},key=lambda w:hashlib.sha256(('glyphdex42'+w).encode()).hexdigest())
    if len(writers)<3:
        raise ValueError('Collect at least three distinct writers for separate train, validation and test groups. Do not invent writers by splitting one person into groups.')
    n=max(1,len(writers)//5)
    for i,writer in enumerate(writers):
        writer_splits[writer]='test' if i<n else 'validation' if i<2*n else 'train'
    output={}
    summary={'note':'Writer groups must represent real distinct writers; browser identifiers alone cannot guarantee this.', 'writers':writer_splits}
    for split in ['train','validation','test']:
        selected=[r for r in records if writer_splits[r[0]]==split]
        output[split+'_images']=np.stack([r[2] for r in selected])
        output[split+'_labels']=np.array([r[1] for r in selected],dtype=np.int64)
        summary[split]={'examples':len(selected),'writers':len({r[0] for r in selected})}
    return output,summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exports',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,default=Path('website_corrections.npz'))
    args=parser.parse_args()
    labels=json.loads((Path(__file__).parent/'models/characters/labels.json').read_text(encoding='utf-8'))
    arrays,summary=prepare_exports(args.exports,labels)
    np.savez_compressed(args.output,**arrays)
    args.output.with_suffix('.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
