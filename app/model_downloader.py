from __future__ import annotations
import hashlib, os, time, urllib.request
from pathlib import Path
from typing import Any

FIRE_SHA='b66acda95a5eaf5154031cf80c1ed906b8cb7838617424afa9b837c5cd44064a'
FIRE_URLS=[
 'https://huggingface.co/fireviewer/litert-models/resolve/330e9097409042751988e9fa5994b51ac2b577bc/models/fireviewer_dfine_m_strict_v1_learning/model.tflite',
 'https://huggingface.co/fireviewer/litert-models/resolve/main/models/fireviewer_dfine_m_strict_v1_learning/model.tflite',
]

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()

def _good(path:Path,expected:str,min_size:int)->bool:
    try:return path.exists() and path.stat().st_size>=min_size and sha256(path)==expected
    except Exception:return False

def _download(urls:list[str],target:Path,expected:str,timeout=240)->dict[str,Any]:
    target.parent.mkdir(parents=True,exist_ok=True);errors=[]
    for url in urls:
        tmp=target.with_suffix(target.suffix+'.download');tmp.unlink(missing_ok=True)
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 Hesham-Fire-Smoke-AI-DFINE','Accept':'application/octet-stream,*/*'})
            with urllib.request.urlopen(req,timeout=timeout) as resp,tmp.open('wb') as out:
                total=0
                while True:
                    c=resp.read(1024*1024)
                    if not c:break
                    out.write(c);total+=len(c)
                    if total and total%(10*1024*1024)<len(c):print(f'  {target.name}: {total/(1024*1024):.1f} MB',flush=True)
            got=sha256(tmp)
            if got!=expected:raise RuntimeError(f'SHA-256 mismatch: {got}')
            os.replace(tmp,target);return {'ok':True,'path':str(target),'url':url,'sha256':got}
        except Exception as e:
            errors.append(f'{url}: {e}');tmp.unlink(missing_ok=True);time.sleep(.5)
    return {'ok':False,'error':' | '.join(errors)}

def validate_models(root:str|Path)->dict[str,Any]:
    root=Path(root);model=root/'fire'/'model.tflite';ok=_good(model,FIRE_SHA,50_000_000)
    return {'dfine':ok,'fire':ok,'smoke':ok,'engine':'D-FINE-M only'}

def download_models(root:str|Path,force=False)->dict[str,Any]:
    root=Path(root);fd=root/'fire';fd.mkdir(parents=True,exist_ok=True);model=fd/'model.tflite'
    if force or not _good(model,FIRE_SHA,50_000_000):
        model.unlink(missing_ok=True);res=_download(FIRE_URLS,model,FIRE_SHA)
    else:
        res={'ok':True,'cached':True,'path':str(model)}
    return {'ok':bool(res.get('ok')),'dfine':res,'fire':res,'smoke':res,'engine':'D-FINE-M only'}
