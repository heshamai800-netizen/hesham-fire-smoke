from __future__ import annotations
import threading
from pathlib import Path
from typing import Any
from .camera_worker import CameraWorker

class CameraManager:
    def __init__(self,db,detector,events_root:str|Path):
        self.db=db;self.detector=detector;self.events_root=Path(events_root);self.workers={};self.lock=threading.RLock()
    def _on_event(self,camera_id,camera_name,kind,label,confidence,snapshot_path):
        self.db.add_event(camera_id,camera_name,kind,label,confidence,snapshot_path)
    def _make_worker(self,cfg):return CameraWorker(cfg,self.detector,self.events_root,self._on_event)
    def sync_from_db(self):
        configs=self.db.list_cameras();ids={c['id'] for c in configs}
        with self.lock:
            for cid in list(self.workers):
                if cid not in ids:
                    w=self.workers.pop(cid);threading.Thread(target=w.stop,daemon=True,name='stop-'+cid).start()
            for cfg in configs:self.apply_saved_config(cfg)
    def apply_saved_config(self,cfg:dict[str,Any]):
        cid=cfg['id'];enabled=bool(cfg.get('enabled',1))
        with self.lock:
            w=self.workers.get(cid)
            if w is None:
                w=self._make_worker(cfg);self.workers[cid]=w
                if enabled:w.start()
                return None
            old_source=w.config.get('source');old_type=w.config.get('source_type');was=bool(w.config.get('enabled',1))
            source_changed=(old_source!=cfg.get('source') or old_type!=cfg.get('source_type'));w.update_config(cfg)
            if source_changed:
                def restart_worker():
                    w.stop()
                    if bool(w.config.get('enabled',1)):w.start()
                threading.Thread(target=restart_worker,daemon=True,name='source-restart-'+cid).start()
            elif enabled and not was:w.start()
            elif not enabled and was:threading.Thread(target=w.stop,daemon=True,name='disable-'+cid).start()
        return None
    def restart(self,cid):
        with self.lock:
            w=self.workers.get(cid)
            if not w:return False
            def do_restart():
                w.stop()
                if bool(w.config.get('enabled',1)):w.start()
            threading.Thread(target=do_restart,daemon=True,name='manual-restart-'+cid).start();return True
    def remove(self,cid):
        with self.lock:w=self.workers.pop(cid,None)
        if w:threading.Thread(target=w.stop,daemon=True,name='delete-'+cid).start()
    def stop_all(self):
        with self.lock:workers=list(self.workers.values())
        for w in workers:w.stop()
    def states(self):
        with self.lock:return {cid:w.snapshot_state() for cid,w in self.workers.items()}
    def get_jpeg_packet(self,cid):
        with self.lock:w=self.workers.get(cid)
        return w.get_jpeg_packet() if w else (0,None)
    def get_config(self,cid):
        with self.lock:w=self.workers.get(cid);return dict(w.config) if w else None
