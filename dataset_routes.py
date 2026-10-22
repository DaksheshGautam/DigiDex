"""Owner-only dataset endpoints. Model predictions never become labels automatically."""
import hashlib
import hmac
import os
import secrets
import threading
import time
from collections import defaultdict,deque
from datetime import timedelta
from flask import Blueprint,current_app,jsonify,request,session
from dataset_store import DatasetStore,DatasetValidationError,validate_sample

dataset=Blueprint('dataset',__name__,url_prefix='/dataset')
attempts=defaultdict(deque);attempt_lock=threading.Lock()


def configured():
    return bool(current_app.config.get('DATASET_KEY_HASH') and current_app.secret_key and current_app.extensions.get('dataset_store'))


def key_matches(value):
    if not isinstance(value,str) or not 20<=len(value)<=256:return False
    return hmac.compare_digest(hashlib.sha256(value.encode()).hexdigest(),current_app.config.get('DATASET_KEY_HASH',''))


def bearer_authorized():
    value=request.headers.get('Authorization','')
    return value.startswith('Bearer ') and key_matches(value[7:])


def authorized(write=False):
    if not configured():return False
    if bearer_authorized():return True
    if not session.get('dataset_owner'):return False
    if write:
        token=request.headers.get('X-Dataset-CSRF','')
        if not token or not hmac.compare_digest(token,session.get('dataset_csrf','')):return False
    return True


def same_origin():
    origin=request.headers.get('Origin')
    # Browser requests must originate on this app. Bearer-authenticated scripts
    # do not rely on cookies and may omit Origin.
    if not origin:return request.headers.get('Authorization','').startswith('Bearer ') or current_app.testing
    from urllib.parse import urlsplit
    return urlsplit(origin).netloc==request.host and urlsplit(origin).scheme=='https'


@dataset.before_request
def protect():
    if request.method!='GET' and not same_origin():
        return jsonify(error='Open the dataset controls on the GlyphDex website.'),403


@dataset.after_request
def no_cache(response):
    response.headers['Cache-Control']='no-store, private'
    response.headers['X-Content-Type-Options']='nosniff'
    return response


@dataset.get('/status')
def status():
    result={'configured':configured(),'authenticated':authorized()}
    if result['authenticated']:
        result['csrf']=session.get('dataset_csrf','')
        try:
            result.update(current_app.extensions['dataset_store'].counts())
            result['csrf']=session.get('dataset_csrf','')
        except Exception:
            result['error']='The private dataset is temporarily unavailable. Corrections will wait in the local queue.'
    return jsonify(result)


@dataset.post('/session')
def login():
    if not configured():return jsonify(error='Private online storage has not been configured.'),503
    # Keep only short-lived in-memory rate limits; do not store IP addresses in the dataset.
    address=request.remote_addr or 'unknown';now=time.monotonic()
    with attempt_lock:
        bucket=attempts[address]
        while bucket and bucket[0]<now-60:bucket.popleft()
        if len(bucket)>=6:return jsonify(error='Please wait a minute before trying another access key.'),429
        bucket.append(now)
        if len(attempts)>1000:
            for key in list(attempts):
                if not attempts[key] or attempts[key][-1]<now-60:attempts.pop(key,None)
    payload=request.get_json(silent=True) or {}
    if not isinstance(payload,dict) or not key_matches(payload.get('key')):
        return jsonify(error='The private dataset access key is incorrect.'),401
    session.clear();session.permanent=True
    session['dataset_owner']=True;session['dataset_csrf']=secrets.token_urlsafe(24)
    return jsonify(authenticated=True,csrf=session['dataset_csrf'])


@dataset.post('/logout')
def logout():
    if not authorized(write=True):return jsonify(error='Connect the private dataset first.'),401
    session.clear();return jsonify(authenticated=False)


@dataset.post('/samples')
def save():
    if not authorized(write=True):return jsonify(error='Connect the private dataset before syncing corrections.'),401
    payload=request.get_json(silent=True)
    if not isinstance(payload,dict) or not isinstance(payload.get('samples'),list) or not 1<=len(payload['samples'])<=32:
        return jsonify(error='Send between 1 and 32 corrected examples.'),400
    try:
        items=[validate_sample(item,current_app.config['DATASET_LABELS']) for item in payload['samples']]
        counts=current_app.extensions['dataset_store'].put(items)
        return jsonify(saved=len(items),**counts)
    except DatasetValidationError as exc:return jsonify(error=str(exc)),400
    except Exception:
        return jsonify(error='Online saving is temporarily unavailable. The correction is queued in this browser.'),503


@dataset.get('/export')
def export():
    if not authorized():return jsonify(error='Private dataset access is required.'),401
    try:
        limit=int(request.args.get('limit',200))
        if not 1<=limit<=500:raise DatasetValidationError('Invalid page size.')
        result=current_app.extensions['dataset_store'].export_page(request.args.get('cursor',''),limit)
        return jsonify(result)
    except (ValueError,DatasetValidationError) as exc:return jsonify(error=str(exc)),400
    except Exception:return jsonify(error='The dataset export is temporarily unavailable.'),503


def install_dataset(app,labels):
    app.config.update(DATASET_KEY_HASH=os.getenv('GLYPHDEX_DATASET_KEY_HASH',''),DATASET_LABELS=labels,
                      SESSION_COOKIE_NAME='glyphdex_dataset',SESSION_COOKIE_SECURE=True,
                      SESSION_COOKIE_HTTPONLY=True,SESSION_COOKIE_SAMESITE='Strict',
                      PERMANENT_SESSION_LIFETIME=timedelta(days=7))
    app.secret_key=os.getenv('GLYPHDEX_DATASET_SESSION_SECRET')
    url=os.getenv('GLYPHDEX_DATABASE_URL','');ca=os.getenv('GLYPHDEX_DATABASE_CA','')
    if url and ca:app.extensions['dataset_store']=DatasetStore(url,ca)
    app.register_blueprint(dataset)
