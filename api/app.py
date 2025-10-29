from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import asyncio
import json
from datetime import datetime

from config.settings import settings
from src.stream import StreamHandler
from src.database import DatabaseManager
import uvicorn


app = FastAPI(
    title = settings.PROJECT_NAME,
    version = settings.VERSION,
    description = "API para la detección de personas en tiempo real",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

stream_handler: StreamHandler = None

def serialize_for_json(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def get_stream_handler():
    global stream_handler;
    if stream_handler is None:
        stream_handler = StreamHandler(use_database = True)
        stream_handler.start()
    return stream_handler


@app.on_event("startup")
async def startup_event():
    print("=" * 70)
    print(f"INICIANDO {settings.PROJECT_NAME} v{settings.VERSION}")
    print("=" * 70)
    
    get_stream_handler()
    
    print(f"API disponible en: http://{settings.API_HOST}:{settings.API_PORT}")
    print("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    global stream_handler
    if stream_handler:
        stream_handler.stop()
    
    print("=" * 70)
    print(f"{settings.PROJECT_NAME} DETENIDO")
    print("=" * 70)

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("Home.html", {"request": request})

@app.get("/health")
async def health_check():
    handler = get_stream_handler()
    return {
        "status": "health",
        "version": settings.VERSION,
        "stream_active": handler.is_alive()
    }

def generate_frames():
    handler = get_stream_handler()

    while True:
        frame_bytes = handler.get_jpeg_frame()

        if frame_bytes is None:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/stats")
async def stats():
    handler = get_stream_handler()
    
    stats = handler.get_statistics()

    return JSONResponse(
        content = json.loads(json.dumps(stats, default=serialize_for_json))
    )

@app.get("/api/reset")
async def reset_stats():
    handler = get_stream_handler()
    handler.reset_counter()
    return {
        "status": "success",
        "message": "Contadores reseteados"
        }

@app.get("/api/recent_entries")
async def get_recent_entries():
    try:
        db = DatabaseManager()
        entries = db.get_recent_entries()
        db.close()
        return entries
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.get("/api/peak_hours")
async def get_peak_hours():
    try:
        db = DatabaseManager()
        results = db.get_peak_hours()
        db.close()
        
        for row in results:
            if 'timestamp' in row and row['timestamp']:
                row['timestamp'] = row['timestamp']
            if 'resultado' in row:
                row['resultado'] = json.loads(row['resultado']) if isinstance(row['resultado'], str) else row['resultado']
        
        return results
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

  
@app.get("/api/weather_predictions")
async def get_weather_predictions(limit: int = 1):
    try:
        db = DatabaseManager()
        results = db.get_weather_predictions(limit)
        db.close()
        
        for row in results:
            if 'timestamp' in row and row['timestamp']:
                row['timestamp'] = row['timestamp']
            if 'resultado' in row:
                row['resultado'] = json.loads(row['resultado']) if isinstance(row['resultado'], str) else row['resultado']
        
        return results
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.get("/api/predictions")
async def get_predictions(limit: int = 1):
    try:
        db = DatabaseManager()
        results = db.get_predictions(limit)
        db.close()
        
        for row in results:
            if 'timestamp' in row and row['timestamp']:
                row['timestamp'] = row['timestamp']
            if 'resultado' in row:
                row['resultado'] = json.loads(row['resultado']) if isinstance(row['resultado'], str) else row['resultado']
        
        return results
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    await websocket.accept()
    handler = get_stream_handler()

    try:
        while True:
            stats = handler.get_statistics()

            stats_json = json.dumps(stats, default=serialize_for_json)
            await websocket.send_text(stats_json)

            await asyncio.sleep(2)

    except WebSocketDisconnect:
        print("Cliente desconectado")
    except Exception as e:
        print(f"Error en el WebSocket: {str(e)}")

if __name__ == "__main__":
    
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False
    )