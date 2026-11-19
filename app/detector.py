from __future__ import annotations
from pathlib import Path
from .fire_detector import DFineFireSmokeCore,DFineClassDetector

class UnifiedDetector:
    def __init__(self,models_root:str|Path):
        root=Path(models_root)
        self.core=DFineFireSmokeCore(root/'fire')
        self.fire=DFineClassDetector(self.core,1,'fire')
        self.smoke=DFineClassDetector(self.core,0,'smoke')
    @property
    def ready(self): return self.core.ready
    @property
    def partial_ready(self): return self.core.ready
    def reload(self): self.core.reload()
    def info(self):
        return {
            'ready':self.ready,
            'partial_ready':self.partial_ready,
            'engine':'D-FINE-M only',
            'fire':self.fire.info(),
            'smoke':self.smoke.info(),
        }
