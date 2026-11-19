from __future__ import annotations
import os, threading, time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import cv2

class CameraWorker:
    PREVIEW_FPS=15.0
    PREVIEW_MAX_WIDTH=1280

    def __init__(self,config:dict[str,Any],detector,events_root:str|Path,on_event:Callable):
        self.config=dict(config);self.detector=detector;self.events_root=Path(events_root);self.on_event=on_event
        self.stop_event=threading.Event();self.capture_thread=None;self.preview_thread=None;self.ai_thread=None
        self.frame_lock=threading.Lock();self.latest_frame=None
        self.jpeg_lock=threading.Lock();self.latest_jpeg=None;self.jpeg_seq=0
        self.cap_lock=threading.Lock();self._cap=None
        self.clip_lock=threading.Lock();self.clip_buffer=deque(maxlen=1);self.last_sample_mono=0.0
        self.det_lock=threading.Lock();self.fire_dets=[];self.smoke_dets=[]
        self.status='STOPPED';self.last_error='';self.capture_fps=0.0;self.preview_fps=0.0;self.fire_fps=0.0;self.smoke_fps=0.0
        self.last_fire_ms=0.0;self.last_smoke_ms=0.0;self.last_frame_at=0.0;self.frames_read=0;self.frames_previewed=0;self.fire_frames=0;self.smoke_frames=0;self.reconnects=0
        self.last_semantic_score=0.0;self.last_proposal_count=0;self.last_top_labels=[];self._last_fire_ai=0.0;self._last_smoke_ai=0.0
        self._positive={'fire':0,'smoke':0};self._positive_since={'fire':0.0,'smoke':0.0};self._clear={'fire':0,'smoke':0};self._latched={'fire':False,'smoke':False};self._last_event={'fire':0.0,'smoke':0.0}

    @property
    def id(self):return self.config['id']

    def start(self):
        if self.capture_thread and self.capture_thread.is_alive():return
        self.stop_event.clear()
        self.capture_thread=threading.Thread(target=self._capture_loop,daemon=True,name='cap-'+self.id)
        self.preview_thread=threading.Thread(target=self._preview_loop,daemon=True,name='preview-'+self.id)
        self.ai_thread=threading.Thread(target=self._ai_loop,daemon=True,name='ai-'+self.id)
        for t in (self.capture_thread,self.preview_thread,self.ai_thread):t.start()

    def stop(self,timeout=3):
        self.stop_event.set()
        with self.cap_lock:cap=self._cap;self._cap=None
        if cap is not None:
            try:cap.release()
            except Exception:pass
        for t in (self.capture_thread,self.preview_thread,self.ai_thread):
            if t and t.is_alive():t.join(timeout=timeout)
        self.status='STOPPED'

    def update_config(self,c):
        old=dict(self.config);self.config=dict(c)
        if any(old.get(k)!=self.config.get(k) for k in ('sample_fps','analysis_width','incense_mode')):
            with self.clip_lock:self.clip_buffer.clear()
            self.last_sample_mono=0.0
        fire_keys=('fire_confidence','fire_confirm_cycles','fire_confirm_sec','fire_enabled')
        smoke_keys=('smoke_confidence','proposal_sensitivity','max_regions','smoke_confirm_cycles','smoke_confirm_sec','smoke_enabled','incense_mode')
        if any(old.get(k)!=self.config.get(k) for k in fire_keys):self._reset_event('fire')
        if any(old.get(k)!=self.config.get(k) for k in smoke_keys):self._reset_event('smoke')

    def _reset_event(self,kind):
        self._positive[kind]=0;self._positive_since[kind]=0.0;self._clear[kind]=0;self._latched[kind]=False

    def snapshot_state(self):
        return {
            'status':self.status,'last_error':self.last_error,'capture_fps':round(self.capture_fps,2),'preview_fps':round(self.preview_fps,2),
            'fire_fps':round(self.fire_fps,2),'smoke_fps':round(self.smoke_fps,2),'last_fire_ms':round(self.last_fire_ms,1),'last_smoke_ms':round(self.last_smoke_ms,1),
            'last_frame_at':self.last_frame_at,'frames_read':self.frames_read,'frames_previewed':self.frames_previewed,'fire_frames':self.fire_frames,'smoke_frames':self.smoke_frames,
            'reconnects':self.reconnects,'fire_enabled':bool(self.config.get('fire_enabled',1)),'smoke_enabled':bool(self.config.get('smoke_enabled',1)),
            'fire_ready':bool(self.detector.fire.ready),'smoke_ready':bool(self.detector.smoke.ready),'engine':'D-FINE-M only',
        }

    def get_jpeg_packet(self):
        with self.jpeg_lock:return self.jpeg_seq,self.latest_jpeg

    def _open(self):
        src=self.config.get('source','0');typ=self.config.get('source_type','usb');source=int(src) if typ=='usb' and str(src).strip().isdigit() else src
        if typ=='usb':
            backend=cv2.CAP_DSHOW if os.name=='nt' and hasattr(cv2,'CAP_DSHOW') else cv2.CAP_ANY
        else:backend=cv2.CAP_FFMPEG
        if typ!='usb':
            # RTSP low-latency options require FFmpeg backend, discard stale frames.
            os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS',
                                  'rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0')
        cap=cv2.VideoCapture(source,backend)
        try:cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        except Exception:pass
        for prop_name,value in (('CAP_PROP_OPEN_TIMEOUT_MSEC',4000),('CAP_PROP_READ_TIMEOUT_MSEC',4000)):
            prop=getattr(cv2,prop_name,None)
            if prop is not None:
                try:cap.set(prop,value)
                except Exception:pass
        return cap

    def _analysis_frame(self,frame):
        w=int(self.config.get('analysis_width',800));h0,w0=frame.shape[:2]
        if w0<=w:return frame.copy()
        s=w/w0;return cv2.resize(frame,(w,max(64,int(h0*s))),interpolation=cv2.INTER_AREA)

    def _capture_loop(self):
        n=0;win=time.monotonic();backoff=float(self.config.get('reconnect_sec',2));next_retry=0.0
        while not self.stop_event.is_set():
            if not bool(self.config.get('enabled',1)):
                self.status='DISABLED';self.stop_event.wait(.15);continue
            if self._cap is None:
                if time.monotonic()<next_retry:self.stop_event.wait(.05);continue
                self.status='CONNECTING'
                try:
                    cap=self._open()
                    if not cap.isOpened():cap.release();raise RuntimeError('unable to open camera source')
                    with self.cap_lock:self._cap=cap
                    self.status='ONLINE';self.last_error='';backoff=float(self.config.get('reconnect_sec',2))
                except Exception as e:
                    self.status='RECONNECTING';self.last_error=f'Camera: {e}';self.reconnects+=1;next_retry=time.monotonic()+backoff;backoff=min(30,max(1,backoff*1.7));continue
            with self.cap_lock:cap=self._cap
            ok,frame=cap.read() if cap is not None else (False,None)
            if not ok or frame is None:
                self.status='RECONNECTING';self.last_error='Camera: frame read failed';self.reconnects+=1
                with self.det_lock:self.fire_dets=[];self.smoke_dets=[]
                with self.cap_lock:old=self._cap;self._cap=None
                try:old.release()
                except Exception:pass
                next_retry=time.monotonic()+float(self.config.get('reconnect_sec',2));continue
            now=time.monotonic();self.frames_read+=1;n+=1;self.last_frame_at=time.time();self.status='ONLINE'
            with self.frame_lock:self.latest_frame=frame
            if now-win>=1:self.capture_fps=n/(now-win);n=0;win=now
        self.status='STOPPED'

    def _preview_loop(self):
        interval=1.0/self.PREVIEW_FPS;n=0;win=time.monotonic();next_at=0.0
        while not self.stop_event.is_set():
            now=time.monotonic()
            if now<next_at:self.stop_event.wait(min(.02,next_at-now));continue
            next_at=now+interval
            with self.frame_lock:frame=None if self.latest_frame is None else self.latest_frame.copy()
            if frame is None:self.stop_event.wait(.03);continue
            self._render(frame);self.frames_previewed+=1;n+=1
            now2=time.monotonic()
            if now2-win>=1:self.preview_fps=n/(now2-win);n=0;win=now2


    @staticmethod
    def _scale_dets(dets,src_shape,dst_shape):
        sh,sw=src_shape[:2];dh,dw=dst_shape[:2];sx=dw/max(1,sw);sy=dh/max(1,sh);out=[]
        for d in dets:
            q=dict(d);x1,y1,x2,y2=d['xyxy'];q['xyxy']=[int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)];out.append(q)
        return out

    def _ai_loop(self):
        """One shared model pass for fire and smoke; never builds a frame backlog."""
        cnt=0;win=time.monotonic();last_run=0.0;last_seq=-1
        while not self.stop_event.is_set():
            fire_on=bool(self.config.get('fire_enabled',1))
            smoke_on=bool(self.config.get('smoke_enabled',1))
            if (not bool(self.config.get('enabled',1)) or not (fire_on or smoke_on)
                    or not self.detector.core.ready):
                with self.det_lock:self.fire_dets=[];self.smoke_dets=[]
                self.stop_event.wait(.10);continue
            # The same model produces both predictions. Choose the shorter requested
            # interval so neither class runs slower than requested, with one pass.
            intervals=[]
            if fire_on:intervals.append(float(self.config.get('fire_interval_sec',.30)))
            if smoke_on:intervals.append(float(self.config.get('smoke_interval_sec',.30)))
            interval=max(.05,min(intervals))
            if time.monotonic()-last_run<interval:
                self.stop_event.wait(.015);continue
            with self.frame_lock:
                seq=self.frames_read
                frame=None if self.latest_frame is None or seq==last_seq else self.latest_frame.copy()
            if frame is None:self.stop_event.wait(.025);continue
            last_seq=seq;last_run=time.monotonic()
            try:
                fire,smoke,ms=self.detector.core.infer_both(
                    frame,
                    float(self.config.get('fire_confidence',.12)),
                    float(self.config.get('smoke_confidence',.20)))
                if not fire_on:fire=[]
                if not smoke_on:smoke=[]
                with self.det_lock:self.fire_dets=fire;self.smoke_dets=smoke
                self.last_fire_ms=ms if fire_on else 0.0
                self.last_smoke_ms=ms if smoke_on else 0.0
                if fire_on:self.fire_frames+=1
                if smoke_on:self.smoke_frames+=1
                cnt+=1
                if fire_on:self._update_event('fire',fire,frame)
                if smoke_on:self._update_event('smoke',smoke,frame)
                if self.last_error.startswith('AI:'):self.last_error=''
            except Exception as e:
                with self.det_lock:self.fire_dets=[];self.smoke_dets=[]
                self.last_error='AI: '+str(e)
            now=time.monotonic()
            if now-win>=1:
                rate=cnt/(now-win)
                self.fire_fps=rate if fire_on else 0.0
                self.smoke_fps=rate if smoke_on else 0.0
                cnt=0;win=now

    def _update_event(self,kind,dets,frame):
        now=time.monotonic();pos=bool(dets)
        if pos:
            if self._positive[kind]==0:self._positive_since[kind]=now
            self._positive[kind]+=1;self._clear[kind]=0
            need=int(self.config.get(f'{kind}_confirm_cycles',2));sec=float(self.config.get(f'{kind}_confirm_sec',.4 if kind=='smoke' else .2))
            if not self._latched[kind] and self._positive[kind]>=need and now-self._positive_since[kind]>=sec:
                self._latched[kind]=True
                if frame is not None and now-self._last_event[kind]>=float(self.config.get('event_cooldown_sec',30)):
                    best=max(dets,key=lambda x:x.get('confidence',0));self._save_event(kind,frame,best);self._last_event[kind]=now
        else:
            self._positive[kind]=0;self._positive_since[kind]=0.0;self._clear[kind]+=1
            if self._clear[kind]>=int(self.config.get('clear_cycles',3)):self._latched[kind]=False

    def _save_event(self,kind,frame,det):
        out=frame.copy();self._draw_box(out,det,kind)
        now=datetime.now();folder=self.events_root/now.strftime('%Y-%m-%d');folder.mkdir(parents=True,exist_ok=True)
        path=folder/f"{now.strftime('%H%M%S_%f')}_{self.id}_{kind}.jpg";cv2.imwrite(str(path),out,[cv2.IMWRITE_JPEG_QUALITY,94])
        self.on_event(self.id,self.config.get('name','Camera'),kind,kind,float(det.get('confidence',0)),str(path))

    def _draw_box(self,frame,d,kind):
        h,w=frame.shape[:2];x1,y1,x2,y2=[int(v) for v in d['xyxy']];x1=max(0,min(w-2,x1));x2=max(x1+1,min(w-1,x2));y1=max(0,min(h-2,y1));y2=max(y1+1,min(h-1,y2))
        color=(0,0,255) if kind=='fire' else (0,200,255)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,0),9,cv2.LINE_AA);cv2.rectangle(frame,(x1,y1),(x2,y2),color,5,cv2.LINE_AA)
        L=max(24,min(70,int(min(x2-x1,y2-y1)*.25)))
        for p1,p2 in [((x1,y1),(x1+L,y1)),((x1,y1),(x1,y1+L)),((x2,y1),(x2-L,y1)),((x2,y1),(x2,y1+L)),((x1,y2),(x1+L,y2)),((x1,y2),(x1,y2-L)),((x2,y2),(x2-L,y2)),((x2,y2),(x2,y2-L))]:cv2.line(frame,p1,p2,(255,255,255),3,cv2.LINE_AA)
        model='D-FINE-M'
        text=f"{kind.upper()} {float(d.get('confidence',0))*100:.1f}%  {model}";scale=max(.6,min(1.0,w/1400));(tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,2);top=max(0,y1-th-18)
        cv2.rectangle(frame,(x1,top),(min(w-1,x1+tw+18),top+th+16),(0,0,0),-1);cv2.rectangle(frame,(x1,top),(min(w-1,x1+tw+18),top+th+16),color,2);cv2.putText(frame,text,(x1+8,top+th+5),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)

    def _render(self,frame):
        src_shape=frame.shape
        if frame.shape[1]>self.PREVIEW_MAX_WIDTH:
            s=self.PREVIEW_MAX_WIDTH/frame.shape[1];out=cv2.resize(frame,(self.PREVIEW_MAX_WIDTH,max(64,int(frame.shape[0]*s))),interpolation=cv2.INTER_AREA)
        else:out=frame
        with self.det_lock:fire=[dict(d) for d in self.fire_dets];smoke=[dict(d) for d in self.smoke_dets]
        if out.shape[:2]!=src_shape[:2]:fire=self._scale_dets(fire,src_shape,out.shape);smoke=self._scale_dets(smoke,src_shape,out.shape)
        for d in fire:self._draw_box(out,d,'fire')
        for d in smoke:self._draw_box(out,d,'smoke')
        missing=[]
        if bool(self.config.get('fire_enabled',1)) and not self.detector.fire.ready:missing.append('FIRE MODEL NOT READY')
        if bool(self.config.get('smoke_enabled',1)) and not self.detector.smoke.ready:missing.append('SMOKE MODEL NOT READY')
        if missing:
            cv2.rectangle(out,(0,0),(out.shape[1],38),(0,0,150),-1);cv2.putText(out,' | '.join(missing),(10,26),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
        txt=f"{self.config.get('name','Camera')} | {self.status} | D-FINE-M | Fire {'ON' if self.config.get('fire_enabled',1) else 'OFF'} | Smoke {'ON' if self.config.get('smoke_enabled',1) else 'OFF'}"
        cv2.rectangle(out,(0,max(0,out.shape[0]-28)),(out.shape[1],out.shape[0]),(0,0,0),-1);cv2.putText(out,txt,(10,out.shape[0]-8),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
        ok,j=cv2.imencode('.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,78])
        if ok:
            with self.jpeg_lock:self.latest_jpeg=j.tobytes();self.jpeg_seq+=1
