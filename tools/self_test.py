from __future__ import annotations
import sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.database import Database
from app.schemas import CameraConfig

def main():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'test.db';db=Database(p)
        cfg=CameraConfig(name='Offline RTSP',source_type='rtsp',source='rtsp://127.0.0.1:65530/live').model_dump();cfg['id']='cam1'
        a=db.create_camera(cfg)
        assert abs(float(a['fire_confidence'])-.12)<1e-6 and abs(float(a['smoke_confidence'])-.20)<1e-6
        cfg['fire_confidence']=.57;cfg['smoke_confidence']=.41
        b=db.update_camera('cam1',cfg)
        assert int(b['settings_revision'])==2 and abs(float(b['fire_confidence'])-.57)<1e-6 and abs(float(b['smoke_confidence'])-.41)<1e-6
        db.add_event('cam1','Offline RTSP','fire','fire',.88,str(Path(td)/'fire.jpg'))
        db.add_event('cam1','Offline RTSP','smoke','smoke',.77,str(Path(td)/'smoke.jpg'))
        ev=db.list_events();assert {x['kind'] for x in ev}=={'fire','smoke'}
        del db
        p.rename(Path(td)/'renamed.db')
    print('SELF_TEST_OK')
    return 0
if __name__=='__main__':raise SystemExit(main())
