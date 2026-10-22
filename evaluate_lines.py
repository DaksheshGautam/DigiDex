"""Evaluate explicit whole-line ground truth through the complete web pipeline.

Use --split-report website_corrections.json to evaluate only held-out test
writers. Without that report, this is an exploratory check, not a benchmark.
No training or automatic uploads take place.
"""
import argparse
import base64
import io
import json
from pathlib import Path
from PIL import Image


def evaluate(paths, client, labels, split_report=None):
    allowed_modes={'mixed','digits','uppercase','lowercase','symbols'}
    test_writers=None
    if split_report is not None:
        report=json.loads(Path(split_report).read_text(encoding='utf-8'))
        test_writers={w for w,s in report['writers'].items() if s=='test'}
        if not test_writers:
            raise ValueError('No test writers in the split report.')
    by_mode={}
    results=[]
    seen=set()
    for path in paths:
        payload=json.loads(Path(path).read_text(encoding='utf-8'))
        if payload.get('schema_version')!=1:
            raise ValueError('Unsupported export schema.')
        for line in payload.get('lines',[]):
            if test_writers is not None and line.get('writer_group') not in test_writers:
                continue
            expected=line.get('text','')
            if (not expected or expected.strip()!=expected or len(expected.replace(' ',''))>32
                    or any(c not in labels and c!=' ' for c in expected)):
                raise ValueError('Invalid whole-line ground truth.')
            mode=line.get('mode')
            uri=line.get('image','')
            if mode not in allowed_modes or not isinstance(uri,str) or not uri.startswith('data:image/png;base64,') or len(uri)>1500000:
                raise ValueError('Invalid test image or recognition mode.')
            with Image.open(io.BytesIO(base64.b64decode(uri.split(',',1)[1],validate=True))) as image:
                if max(image.size)>1024:
                    raise ValueError('Test drawing too large.')
            key=(uri,mode)
            if key in seen:
                raise ValueError('Duplicate test drawing. Remove duplicates before evaluation.')
            seen.add(key)
            response=client.post('/predict',json={'image':uri,'mode':mode})
            result=response.get_json() or {}
            exact=response.status_code==200 and result['text']==expected
            # Count full-pipeline failures too; do not discard segmentation errors.
            uncertain=response.status_code==200 and any(c['uncertain'] for c in result['characters'])
            entry=by_mode.setdefault(mode,{'lines':0,'exact_matches':0,'needs_confirmation':0,'errors':0})
            entry['lines']+=1;entry['exact_matches']+=int(exact)
            entry['needs_confirmation']+=int(uncertain);entry['errors']+=int(response.status_code!=200)
            results.append({'mode':mode,'expected':expected,'reading':result.get('text'),
                            'exact_match':exact,'needs_confirmation':uncertain,'error':result.get('error')})
    for entry in by_mode.values():
        entry['exact_line_accuracy']=entry['exact_matches']/entry['lines']
    return {'held_out_writers':test_writers is not None,'note':'Whole-line checks including spaces, segmentation and rejection. Exploratory unless real writer separation was verified.',
            'by_mode':by_mode,'results':results}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exports',nargs='+',type=Path)
    parser.add_argument('--split-report',type=Path)
    parser.add_argument('--output',type=Path,default=Path('website_line_evaluation.json'))
    args=parser.parse_args()
    from app import app,labels
    result=evaluate(args.exports,app.test_client(),labels,args.split_report)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result['by_mode'],indent=2))
