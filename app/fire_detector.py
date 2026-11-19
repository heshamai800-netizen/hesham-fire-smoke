from __future__ import annotations
import hashlib, threading, time, os, traceback
from pathlib import Path
import cv2
import numpy as np

MODEL_SHA256='b66acda95a5eaf5154031cf80c1ed906b8cb7838617424afa9b837c5cd44064a'
CLASS_NAMES={0:'smoke',1:'fire'}


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def _iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(1,(a[2]-a[0])*(a[3]-a[1]));bb=max(1,(b[2]-b[0])*(b[3]-b[1]))
    return inter/max(1,aa+bb-inter)


def _nms(items,thr=.60):
    kept=[]
    for d in sorted(items,key=lambda x:float(x.get('confidence',0)),reverse=True):
        if any(_iou(d['xyxy'],k['xyxy'])>=thr for k in kept):
            continue
        kept.append(dict(d))
    return kept


def _sigmoid(x):
    x=np.clip(x,-50,50)
    return 1/(1+np.exp(-x))


def _decode(outputs,w,h,size,class_id,threshold):
    det=outputs.get('detections'); rows=None
    if det is not None:
        arr=np.asarray(det,dtype=np.float32)
        if arr.ndim==3 and arr.shape[0]==1: arr=arr[0]
        if arr.ndim==2 and (arr.shape[0]==6 or (arr.shape[0]<=16 and arr.shape[0]<arr.shape[1])): arr=arr.T
        if arr.ndim==2 and arr.shape[1]>=6: rows=arr[:,:6]
    if rows is None:
        boxes=np.asarray(outputs.get('boxes')) if outputs.get('boxes') is not None else None
        logits=np.asarray(outputs.get('logits')) if outputs.get('logits') is not None else None
        if boxes is None or logits is None:
            raise RuntimeError('D-FINE outputs missing detections/boxes/logits')
        if boxes.ndim==3: boxes=boxes[0]
        if logits.ndim==3: logits=logits[0]
        scores=logits.astype(np.float32)
        if float(np.nanmin(scores))<0 or float(np.nanmax(scores))>1: scores=_sigmoid(scores)
        rows=np.concatenate([boxes.astype(np.float32),scores[:,:2]],axis=1)
    sx=w/float(size); sy=h/float(size); out=[]
    for row in rows:
        if len(row)<6 or not np.all(np.isfinite(row[:6])): continue
        c0,c1=float(row[4]),float(row[5])
        cls=0 if c0>=c1 else 1
        score=max(c0,c1)
        if cls!=int(class_id) or score<float(threshold): continue
        cx,cy,bw,bh=[float(v) for v in row[:4]]
        if max(abs(cx),abs(cy),abs(bw),abs(bh))<=2:
            cx*=size;cy*=size;bw*=size;bh*=size
        x1=max(0,min(w-1,(cx-bw/2)*sx));x2=max(0,min(w-1,(cx+bw/2)*sx))
        y1=max(0,min(h-1,(cy-bh/2)*sy));y2=max(0,min(h-1,(cy+bh/2)*sy))
        if x2<=x1 or y2<=y1 or (x2-x1)*(y2-y1)<9: continue
        out.append({
            'xyxy':[int(x1),int(y1),int(x2),int(y2)],
            'confidence':float(score),
            'label':CLASS_NAMES[int(class_id)],
            'source':'D-FINE-M',
            'class_id':int(class_id),
        })
    return _nms(out)


class DFineFireSmokeCore:
    def __init__(self,model_dir:str|Path):
        self.model_dir=Path(model_dir)
        self.model_path=self.model_dir/'model.tflite'
        self.interpreter=None;self.runner=None;self.compiled=None;self.error='';self.gpu_error='';self.provider='none';self.lock=threading.RLock();self.image_size=704;self.loaded_at=0.0
        self._load()

    @property
    def ready(self): return self.runner is not None

    def _load(self):
        self.interpreter=None;self.compiled=None;self.runner=None;self.error='';self.gpu_error='';self.provider='none'
        if not self.model_path.exists():
            self.error='D-FINE-M model missing';return
        try:
            if self.model_path.stat().st_size<50_000_000:
                raise RuntimeError('D-FINE-M model file is too small')
            if sha256_file(self.model_path)!=MODEL_SHA256:
                raise RuntimeError('D-FINE-M model SHA-256 mismatch')
            from ai_edge_litert.interpreter import Interpreter
            # Benchmark actual performance before increasing this value. Too many
            # inference threads can starve capture/preview threads on Windows.
            threads=max(1,min(8,int(os.getenv('HESHAM_CPU_THREADS','4'))))
            it=Interpreter(model_path=str(self.model_path),num_threads=threads)
            it.allocate_tensors()
            sigs=it.get_signature_list() or {}
            if 'infer' not in sigs:raise RuntimeError('D-FINE-M model has no infer signature')
            self.interpreter=it
            self.runner=it.get_signature_runner('infer')
            self.provider=f'LiteRT-CPU ({threads} threads)'
            self.loaded_at=time.time()
        except Exception as e:
            self.error=f'D-FINE-M load failed: {e}'

    def reload(self):
        with self.lock:
            self.interpreter=None;self.runner=None
        self._load()

    def infer_both(self,frame_bgr:np.ndarray,fire_confidence:float,smoke_confidence:float):
        """One model pass -> BOTH class outputs, no second redundant inference."""
        if not self.ready:return [],[],0.0
        start=time.perf_counter();h,w=frame_bgr.shape[:2]
        # Efficient HWC uint8 resize before one float conversion.
        resized=cv2.resize(frame_bgr,(self.image_size,self.image_size),interpolation=cv2.INTER_LINEAR)
        rgb=cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
        x=np.ascontiguousarray(np.transpose(rgb,(2,0,1)),dtype=np.float32)
        x*=1.0/255.0
        with self.lock:
            outputs=self.runner(x=x[None,...])
        ms=(time.perf_counter()-start)*1000.0
        fire=_decode(outputs,w,h,self.image_size,1,float(fire_confidence))
        smoke=_decode(outputs,w,h,self.image_size,0,float(smoke_confidence))
        return fire,smoke,ms

    def infer_class(self,frame_bgr:np.ndarray,class_id:int,confidence:float):
        # Legacy API retained for diagnostic and older external clients.
        f,s,ms=self.infer_both(frame_bgr,confidence,confidence)
        return (f if int(class_id)==1 else s),ms

    def info(self):
        return {
            'ready':self.ready,
            'error':self.error,
            'variant':'D-FINE-M Fire + Smoke',
            'provider':self.provider,
            'device':'CPU',
            'gpu_error':self.gpu_error,
            'path':str(self.model_path),
            'image_size':self.image_size,
            'loaded_at':self.loaded_at,
            'classes':{'0':'smoke_visible','1':'flame_visible'},
        }


class DFineClassDetector:
    def __init__(self,core:DFineFireSmokeCore,class_id:int,name:str):
        self.core=core;self.class_id=int(class_id);self.name=str(name)
    @property
    def ready(self): return self.core.ready
    @property
    def error(self): return self.core.error
    def reload(self): self.core.reload()
    def infer(self,frame_bgr:np.ndarray,confidence:float):
        return self.core.infer_class(frame_bgr,self.class_id,confidence)
    def info(self):
        d=self.core.info().copy()
        d['variant']=f'D-FINE-M {self.name.title()} Class'
        d['class_id']=self.class_id
        d['class_name']='smoke_visible' if self.class_id==0 else 'flame_visible'
        return d
