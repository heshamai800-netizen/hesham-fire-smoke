from __future__ import annotations
import os,time,uuid,asyncio
from pathlib import Path
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS','rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0|stimeout;5000000')
from fastapi import FastAPI,HTTPException,Request,status as http_status
from fastapi.responses import FileResponse,JSONResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
from .camera_manager import CameraManager
from .database import Database
from .detector import UnifiedDetector
from .model_downloader import download_models,validate_models
from .schemas import CameraConfig

APP_VERSION='1.5.0';APP_ID='hesham-fire-smoke-dfine-only'
ROOT=Path(__file__).resolve().parents[1];STATIC=ROOT/'static';DATA=ROOT/'data';EVENTS=ROOT/'events';MODELS=ROOT/'models'
for p in (DATA,EVENTS,MODELS):p.mkdir(parents=True,exist_ok=True)
db=Database(DATA/'fire_smoke_ai.db');detector=UnifiedDetector(MODELS);manager=CameraManager(db,detector,EVENTS)
app=FastAPI(title='Hesham Fire & Smoke AI - D-FINE Only',version=APP_VERSION)
app.mount('/static',StaticFiles(directory=STATIC),name='static');app.mount('/events',StaticFiles(directory=EVENTS),name='events')
BOOL_FIELDS=('enabled','fire_enabled','smoke_enabled','incense_mode')
EDIT_FIELDS=('name','source_type','source','enabled','fire_enabled','smoke_enabled','fire_confidence','smoke_confidence','fire_interval_sec','sample_fps','smoke_interval_sec','analysis_width','proposal_sensitivity','max_regions','fire_confirm_cycles','fire_confirm_sec','smoke_confirm_cycles','smoke_confirm_sec','clear_cycles','event_cooldown_sec','reconnect_sec','incense_mode')

def serialize_camera(row):
    if row is None:return None
    r=dict(row)
    for k in BOOL_FIELDS:r[k]=bool(r.get(k,0))
    return r

def verify_saved(requested,saved):
    mm=[]
    for k in EDIT_FIELDS:
        a=requested.get(k);b=saved.get(k)
        if k in BOOL_FIELDS:
            if bool(a)!=bool(b):mm.append(k)
        elif isinstance(a,(int,float)) and not isinstance(a,bool):
            try:
                if abs(float(a)-float(b))>1e-6:mm.append(k)
            except Exception:mm.append(k)
        elif a!=b:mm.append(k)
    return mm

def ensure_persisted(requested,saved_raw):
    saved=serialize_camera(saved_raw);mm=verify_saved(requested,saved)
    if mm:raise HTTPException(500,'SQLite save verification failed for: '+', '.join(mm))
    return saved

def apply_runtime_nonfatal(cfg):
    try:manager.apply_saved_config(cfg);return None
    except Exception as e:return str(e)

@app.middleware('http')
async def no_cache(request:Request,call_next):
    response=await call_next(request)
    if request.url.path.startswith('/api/') or request.url.path=='/' or request.url.path.startswith('/static/'):
        response.headers['Cache-Control']='no-store, no-cache, must-revalidate, max-age=0';response.headers['Pragma']='no-cache';response.headers['Expires']='0'
    response.headers['X-Hesham-App-Version']=APP_VERSION;return response

@app.on_event('startup')
def startup():
    try:manager.sync_from_db()
    except Exception as e:print('Startup camera sync warning:',e)
@app.on_event('shutdown')
def shutdown():manager.stop_all()
@app.get('/')
def index():return FileResponse(STATIC/'index.html')
@app.get('/api/version')
def version():return {'app':APP_ID,'version':APP_VERSION,'root':str(ROOT),'db_path':str(db.path)}
@app.get('/api/status')
def status():return {'ok':True,'version':APP_VERSION,'time':time.time(),'model':detector.info(),'model_files':validate_models(MODELS),'cameras':manager.states()}
@app.get('/api/db-health')
def db_health():
    try:return {'ok':True,'db_path':str(db.path),'camera_count':len(db.list_cameras()),'version':APP_VERSION}
    except Exception as e:raise HTTPException(500,'SQLite health check failed: '+str(e))
@app.get('/api/cameras')
def cameras():
    states=manager.states();rows=[]
    for row in db.list_cameras():
        r=serialize_camera(row);r['runtime']=states.get(r['id'],{});rows.append(r)
    return rows
@app.get('/api/cameras/{camera_id}')
def get_camera(camera_id:str):
    row=db.get_camera(camera_id)
    if not row:raise HTTPException(404,'Camera not found')
    return serialize_camera(row)
@app.post('/api/cameras',status_code=http_status.HTTP_201_CREATED)
def add_camera(payload:CameraConfig):
    d=payload.model_dump();d['id']=uuid.uuid4().hex[:12]
    try:saved=ensure_persisted(d,db.create_camera(d))
    except HTTPException:raise
    except Exception as e:raise HTTPException(500,'Camera configuration could not be saved to SQLite: '+str(e))
    final=ensure_persisted(d,db.get_camera(saved['id']));warning=apply_runtime_nonfatal(final)
    return {'ok':True,'saved':True,'camera':final,'revision':final.get('settings_revision',1),'db_path':str(db.path),'version':APP_VERSION,'runtime_warning':warning}
@app.put('/api/cameras/{camera_id}')
def update_camera(camera_id:str,payload:CameraConfig):
    if not db.get_camera(camera_id):raise HTTPException(404,'Camera not found')
    d=payload.model_dump();d['id']=camera_id
    try:saved=ensure_persisted(d,db.update_camera(camera_id,d))
    except KeyError:raise HTTPException(404,'Camera not found')
    except HTTPException:raise
    except Exception as e:raise HTTPException(500,'Camera settings could not be saved to SQLite: '+str(e))
    final=ensure_persisted(d,db.get_camera(camera_id));warning=apply_runtime_nonfatal(final)
    return {'ok':True,'saved':True,'camera':final,'revision':final.get('settings_revision',1),'updated_at':final.get('updated_at'),'db_path':str(db.path),'version':APP_VERSION,'runtime_warning':warning}
@app.delete('/api/cameras/{camera_id}')
def delete_camera(camera_id:str):
    if not db.get_camera(camera_id):raise HTTPException(404,'Camera not found')
    db.delete_camera(camera_id);manager.remove(camera_id);return {'ok':True}
@app.post('/api/cameras/{camera_id}/restart')
def restart(camera_id:str):
    if not db.get_camera(camera_id):raise HTTPException(404,'Camera not found')
    if not manager.restart(camera_id):manager.apply_saved_config(db.get_camera(camera_id))
    return {'ok':True}
@app.get('/api/cameras/{camera_id}/stream')
async def stream(camera_id:str,request:Request):
    if not db.get_camera(camera_id):raise HTTPException(404,'Camera not found')
    async def gen():
        last=-1
        while True:
            if await request.is_disconnected():break
            seq,jpeg=manager.get_jpeg_packet(camera_id)
            if jpeg is not None and seq!=last:
                last=seq;yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n'
            await asyncio.sleep(.025)
    return StreamingResponse(gen(),media_type='multipart/x-mixed-replace; boundary=frame',headers={'Cache-Control':'no-store, no-cache, must-revalidate, max-age=0','Pragma':'no-cache','X-Accel-Buffering':'no'})
@app.get('/api/events')
def events(limit:int=100):
    rows=db.list_events(limit)
    for r in rows:
        try:rel=Path(r['snapshot_path']).resolve().relative_to(EVENTS.resolve());r['snapshot_url']='/events/'+str(rel).replace('\\','/')
        except Exception:r['snapshot_url']=''
    return rows
@app.post('/api/model/download')
def model_download():
    result=download_models(MODELS,force=False);detector.reload()
    info=detector.info()
    if not info['fire']['ready'] or not info['smoke']['ready']:
        raise HTTPException(500,'Models repaired/downloaded but runtime is not fully ready. Fire: '+str(info['fire'].get('error',''))+' | Smoke: '+str(info['smoke'].get('error','')))
    return {'ok':True,'download':result,'model':info}
@app.post('/api/model/reload')
def model_reload():detector.reload();return {'ok':detector.partial_ready,'model':detector.info()}
@app.exception_handler(Exception)
async def unhandled(_request,exc:Exception):return JSONResponse(status_code=500,content={'detail':str(exc)})
