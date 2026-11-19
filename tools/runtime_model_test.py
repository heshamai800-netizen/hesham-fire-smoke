from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.detector import UnifiedDetector
from app.model_downloader import validate_models

def main():
    files=validate_models(ROOT/'models');print('Model files:',files)
    if not files['dfine']:
        print('ERROR: Required bundled D-FINE model checksum failed');return 2
    d=UnifiedDetector(ROOT/'models');info=d.info();print('Fire:',info['fire']);print('Smoke:',info['smoke'])
    if not d.fire.ready or not d.smoke.ready:
        print('ERROR: D-FINE runtime not ready:',d.core.error);return 3
    frame=np.zeros((360,640,3),np.uint8)
    try:
        f,s,ms=d.core.infer_both(frame,.99,.99);print('Single D-FINE inference: PASS',round(ms,1),'ms','fire',len(f),'smoke',len(s))
    except Exception as e:
        print('ERROR: D-FINE inference failed:',e);return 4
    print('RUNTIME_MODEL_TEST_OK');return 0
if __name__=='__main__':raise SystemExit(main())
