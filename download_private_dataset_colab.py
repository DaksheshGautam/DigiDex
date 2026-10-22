"""Download private corrections in Colab without saving the access key."""
import getpass
import json
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


def download_dataset(site, key, destination='glyphdex-private-dataset.json'):
    parsed=urlsplit(site)
    if parsed.scheme!='https' or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError('Use the HTTPS address of your GlyphDex website.')
    site=site.rstrip('/')
    output={'schema_version':1,'examples':[],'lines':[]}
    cursor='';seen=set()
    while True:
        request=Request(site+'/dataset/export?'+urlencode({'limit':200,'cursor':cursor}),
                        headers={'Authorization':'Bearer '+key})
        # Do not forward the private key to a redirect destination.
        from urllib.request import HTTPRedirectHandler,build_opener
        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self,*args,**kwargs):return None
        with build_opener(NoRedirect()).open(request,timeout=60) as response:
            page=json.load(response)
        output['examples'].extend(page['examples']);output['lines'].extend(page['lines'])
        cursor=page.get('next_cursor')
        if not cursor:break
        if cursor in seen:raise ValueError('Export cursor repeated; try again later.')
        seen.add(cursor)
    Path(destination).write_text(json.dumps(output,ensure_ascii=False),encoding='utf-8')
    return {'characters':len(output['examples']),'lines':len(output['lines'])}


if __name__=='__main__':
    site=input('GlyphDex HTTPS website: ').strip()
    key=getpass.getpass('Private dataset key (hidden): ')
    try:print(download_dataset(site,key))
    finally:del key
