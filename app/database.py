from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

WRITE_FIELDS = (
    'name','source_type','source','enabled','fire_enabled','smoke_enabled','fire_confidence','smoke_confidence',
    'fire_interval_sec','sample_fps','smoke_interval_sec','analysis_width','proposal_sensitivity','max_regions',
    'fire_confirm_cycles','fire_confirm_sec','smoke_confirm_cycles','smoke_confirm_sec','clear_cycles',
    'event_cooldown_sec','reconnect_sec','incense_mode'
)

class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=20, check_same_thread=False)
        con.row_factory = sqlite3.Row
        try:
            con.execute('PRAGMA journal_mode=WAL')
            con.execute('PRAGMA synchronous=FULL')
            con.execute('PRAGMA busy_timeout=20000')
            yield con
            con.commit()
        except Exception:
            con.rollback(); raise
        finally:
            con.close()

    def _init(self):
        with self._conn() as con:
            con.executescript('''
            CREATE TABLE IF NOT EXISTS cameras(
              id TEXT PRIMARY KEY,name TEXT NOT NULL,source_type TEXT NOT NULL,source TEXT NOT NULL,
              enabled INTEGER NOT NULL DEFAULT 1,fire_enabled INTEGER NOT NULL DEFAULT 1,smoke_enabled INTEGER NOT NULL DEFAULT 1,
              fire_confidence REAL NOT NULL DEFAULT 0.12,smoke_confidence REAL NOT NULL DEFAULT 0.20,
              fire_interval_sec REAL NOT NULL DEFAULT 0.30,sample_fps REAL NOT NULL DEFAULT 8,
              smoke_interval_sec REAL NOT NULL DEFAULT 0.30,analysis_width INTEGER NOT NULL DEFAULT 800,
              proposal_sensitivity REAL NOT NULL DEFAULT 1.45,max_regions INTEGER NOT NULL DEFAULT 5,
              fire_confirm_cycles INTEGER NOT NULL DEFAULT 2,fire_confirm_sec REAL NOT NULL DEFAULT 0.20,
              smoke_confirm_cycles INTEGER NOT NULL DEFAULT 2,smoke_confirm_sec REAL NOT NULL DEFAULT 0.30,
              clear_cycles INTEGER NOT NULL DEFAULT 3,event_cooldown_sec REAL NOT NULL DEFAULT 30,
              reconnect_sec REAL NOT NULL DEFAULT 2,incense_mode INTEGER NOT NULL DEFAULT 1,
              settings_revision INTEGER NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS events(
              id INTEGER PRIMARY KEY AUTOINCREMENT,camera_id TEXT NOT NULL,camera_name TEXT NOT NULL,
              kind TEXT NOT NULL,label TEXT NOT NULL,confidence REAL NOT NULL,snapshot_path TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
            ''')
            cols={r[1] for r in con.execute('PRAGMA table_info(cameras)').fetchall()}
            migrations={
              'enabled':'ALTER TABLE cameras ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1',
              'fire_enabled':'ALTER TABLE cameras ADD COLUMN fire_enabled INTEGER NOT NULL DEFAULT 1',
              'smoke_enabled':'ALTER TABLE cameras ADD COLUMN smoke_enabled INTEGER NOT NULL DEFAULT 1',
              'fire_confidence':'ALTER TABLE cameras ADD COLUMN fire_confidence REAL NOT NULL DEFAULT 0.12',
              'smoke_confidence':'ALTER TABLE cameras ADD COLUMN smoke_confidence REAL NOT NULL DEFAULT 0.20',
              'fire_interval_sec':'ALTER TABLE cameras ADD COLUMN fire_interval_sec REAL NOT NULL DEFAULT 0.30',
              'sample_fps':'ALTER TABLE cameras ADD COLUMN sample_fps REAL NOT NULL DEFAULT 8',
              'smoke_interval_sec':'ALTER TABLE cameras ADD COLUMN smoke_interval_sec REAL NOT NULL DEFAULT 0.30',
              'analysis_width':'ALTER TABLE cameras ADD COLUMN analysis_width INTEGER NOT NULL DEFAULT 800',
              'proposal_sensitivity':'ALTER TABLE cameras ADD COLUMN proposal_sensitivity REAL NOT NULL DEFAULT 1.45',
              'max_regions':'ALTER TABLE cameras ADD COLUMN max_regions INTEGER NOT NULL DEFAULT 5',
              'fire_confirm_cycles':'ALTER TABLE cameras ADD COLUMN fire_confirm_cycles INTEGER NOT NULL DEFAULT 2',
              'fire_confirm_sec':'ALTER TABLE cameras ADD COLUMN fire_confirm_sec REAL NOT NULL DEFAULT 0.20',
              'smoke_confirm_cycles':'ALTER TABLE cameras ADD COLUMN smoke_confirm_cycles INTEGER NOT NULL DEFAULT 2',
              'smoke_confirm_sec':'ALTER TABLE cameras ADD COLUMN smoke_confirm_sec REAL NOT NULL DEFAULT 0.30',
              'clear_cycles':'ALTER TABLE cameras ADD COLUMN clear_cycles INTEGER NOT NULL DEFAULT 3',
              'event_cooldown_sec':'ALTER TABLE cameras ADD COLUMN event_cooldown_sec REAL NOT NULL DEFAULT 30',
              'reconnect_sec':'ALTER TABLE cameras ADD COLUMN reconnect_sec REAL NOT NULL DEFAULT 2',
              'incense_mode':'ALTER TABLE cameras ADD COLUMN incense_mode INTEGER NOT NULL DEFAULT 1',
              'settings_revision':'ALTER TABLE cameras ADD COLUMN settings_revision INTEGER NOT NULL DEFAULT 1',
              'created_at':'ALTER TABLE cameras ADD COLUMN created_at TEXT',
              'updated_at':'ALTER TABLE cameras ADD COLUMN updated_at TEXT',
            }
            for c,sql in migrations.items():
                if c not in cols: con.execute(sql)
            con.execute("UPDATE cameras SET created_at=COALESCE(created_at,CURRENT_TIMESTAMP)")
            con.execute("UPDATE cameras SET updated_at=COALESCE(updated_at,CURRENT_TIMESTAMP)")
            con.execute("UPDATE cameras SET settings_revision=COALESCE(settings_revision,1)")
            ecols={r[1] for r in con.execute('PRAGMA table_info(events)').fetchall()}
            if 'kind' not in ecols: con.execute("ALTER TABLE events ADD COLUMN kind TEXT NOT NULL DEFAULT 'smoke'")

    @staticmethod
    def _values(d: dict[str, Any]):
        return (
            d['name'],d['source_type'],d['source'],int(bool(d.get('enabled',1))),int(bool(d.get('fire_enabled',1))),
            int(bool(d.get('smoke_enabled',1))),float(d.get('fire_confidence',.12)),float(d.get('smoke_confidence',.20)),
            float(d.get('fire_interval_sec',.30)),float(d.get('sample_fps',8)),float(d.get('smoke_interval_sec',.30)),
            int(d.get('analysis_width',800)),float(d.get('proposal_sensitivity',1.45)),int(d.get('max_regions',5)),
            int(d.get('fire_confirm_cycles',2)),float(d.get('fire_confirm_sec',.20)),int(d.get('smoke_confirm_cycles',2)),
            float(d.get('smoke_confirm_sec',.40)),int(d.get('clear_cycles',3)),float(d.get('event_cooldown_sec',30)),
            float(d.get('reconnect_sec',2)),int(bool(d.get('incense_mode',1)))
        )

    def list_cameras(self):
        with self._conn() as con:
            return [dict(r) for r in con.execute("SELECT * FROM cameras ORDER BY COALESCE(created_at,''),id").fetchall()]

    def get_camera(self,cid):
        with self._conn() as con:
            r=con.execute('SELECT * FROM cameras WHERE id=?',(cid,)).fetchone(); return dict(r) if r else None

    def create_camera(self,d):
        cid=d['id']
        with self._conn() as con:
            if con.execute('SELECT 1 FROM cameras WHERE id=?',(cid,)).fetchone(): raise RuntimeError('Camera ID already exists')
            con.execute('''INSERT INTO cameras(
              id,name,source_type,source,enabled,fire_enabled,smoke_enabled,fire_confidence,smoke_confidence,
              fire_interval_sec,sample_fps,smoke_interval_sec,analysis_width,proposal_sensitivity,max_regions,
              fire_confirm_cycles,fire_confirm_sec,smoke_confirm_cycles,smoke_confirm_sec,clear_cycles,
              event_cooldown_sec,reconnect_sec,incense_mode,settings_revision
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)''',(cid,)+self._values(d))
        saved=self.get_camera(cid)
        if saved is None: raise RuntimeError('Camera insert committed but row cannot be read back')
        return saved

    def update_camera(self,cid,d):
        with self._conn() as con:
            cur=con.execute('''UPDATE cameras SET
              name=?,source_type=?,source=?,enabled=?,fire_enabled=?,smoke_enabled=?,fire_confidence=?,smoke_confidence=?,
              fire_interval_sec=?,sample_fps=?,smoke_interval_sec=?,analysis_width=?,proposal_sensitivity=?,max_regions=?,
              fire_confirm_cycles=?,fire_confirm_sec=?,smoke_confirm_cycles=?,smoke_confirm_sec=?,clear_cycles=?,
              event_cooldown_sec=?,reconnect_sec=?,incense_mode=?,settings_revision=COALESCE(settings_revision,0)+1,
              updated_at=CURRENT_TIMESTAMP WHERE id=?''', self._values(d)+(cid,))
            if cur.rowcount != 1: raise KeyError('Camera not found')
        saved=self.get_camera(cid)
        if saved is None: raise RuntimeError('Camera update committed but row cannot be read back')
        return saved

    def delete_camera(self,cid):
        with self._conn() as con: con.execute('DELETE FROM cameras WHERE id=?',(cid,))

    def add_event(self,cid,name,kind,label,confidence,path):
        with self._conn() as con:
            cur=con.execute('INSERT INTO events(camera_id,camera_name,kind,label,confidence,snapshot_path) VALUES(?,?,?,?,?,?)',(cid,name,kind,label,confidence,path)); return int(cur.lastrowid)

    def list_events(self,limit=100):
        limit=max(1,min(int(limit),1000))
        with self._conn() as con:
            return [dict(r) for r in con.execute('SELECT * FROM events ORDER BY id DESC LIMIT ?',(limit,)).fetchall()]
