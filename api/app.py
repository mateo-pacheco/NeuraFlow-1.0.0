from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import uvicorn
import json
from datetime import datetime

import os
from ai_recommendations import RecommendationManager
from config.settings import settings
from src.stream import StreamHandler
from src.database import DatabaseManager

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API para la detección de personas en tiempo real",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://localhost:3000",
        "http://127.0.0.1:4200",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stream_handler: StreamHandler = None


def serialize_for_json(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def get_stream_handler() -> StreamHandler:
    global stream_handler
    if stream_handler is None:
        stream_handler = StreamHandler(use_database=True)
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
    print("API DETENIDA")
    print("=" * 70)


@app.get("/api/health")
async def health_check():
    handler = get_stream_handler()
    return {
        "status": "health",
        "version": settings.VERSION,
        "stream_active": handler.is_alive(),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/info")
async def system_info():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model": settings.MODEL_PATH,
        "camera": settings.CAMERA_SOURCE,
        "database": {
            "host": settings.DB_HOST,
            "name": settings.DB_NAME,
            "connected": True,
        },
    }


def generate_frames():
    handler = get_stream_handler()

    while True:
        frame_bytes = handler.get_jpeg_frame()

        if frame_bytes is None:
            continue

        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.get("/api/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/stats")
async def stats():
    handler = get_stream_handler()

    stats = handler.get_statistics()

    return JSONResponse(
        content=json.loads(json.dumps(stats, default=serialize_for_json))
    )


@app.get("/api/reset")
async def reset_stats():
    handler = get_stream_handler()
    handler.reset_counter()
    return {
        "status": "success",
        "message": "Contadores reseteados",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/recent_entries")
async def get_recent_entries():
    try:
        db = DatabaseManager()
        entries = db.get_recent_entries()
        db.close()
        return entries
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/entries/total")
async def get_total_entries():
    try:
        db = DatabaseManager()
        total = db.get_total_entries()
        db.close()
        return {"total": total}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/peak_hours")
async def get_peak_hours():
    try:
        db = DatabaseManager()
        results = db.get_algorithm_results("peak_hour")
        db.close()

        for row in results:
            if "timestamp" in row and row["timestamp"]:
                row["timestamp"] = row["timestamp"]
            if "resultado" in row:
                row["resultado"] = (
                    json.loads(row["resultado"])
                    if isinstance(row["resultado"], str)
                    else row["resultado"]
                )

        return results
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/weather_predictions")
async def get_weather_predictions():
    try:
        db = DatabaseManager()
        results = db.get_algorithm_results("Weather prediction")
        db.close()

        for row in results:
            if "timestamp" in row and row["timestamp"]:
                row["timestamp"] = row["timestamp"]
            if "resultado" in row:
                row["resultado"] = (
                    json.loads(row["resultado"])
                    if isinstance(row["resultado"], str)
                    else row["resultado"]
                )

        return results
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/predictions")
async def get_predictions():
    try:
        db = DatabaseManager()
        results = db.get_algorithm_results("Prediction")
        db.close()

        for row in results:
            if "timestamp" in row and row["timestamp"]:
                row["timestamp"] = row["timestamp"]
            if "resultado" in row:
                row["resultado"] = (
                    json.loads(row["resultado"])
                    if isinstance(row["resultado"], str)
                    else row["resultado"]
                )

        return results
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    await websocket.accept()
    handler = get_stream_handler()

    try:
        while True:
            stats = handler.get_statistics()

            stats_json = json.dumps(stats, default=serialize_for_json)
            await websocket.send_text(stats_json)

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        print("Cliente desconectado")
    except Exception as e:
        print(f"Error en el WebSocket: {str(e)}")



recommendation_manager = None
if os.getenv("AI_RECOMMENDATIONS_ENABLED", "false").lower() == "true":
    try:
        recommendation_manager = RecommendationManager()
        print("Sistema de recomendaciones IA activado")
    except Exception as e:
        print(f"Recomendaciones IA desactivadas: {e}")


@app.post("/api/recommendations/generate")
async def generate_ai_recommendation():
    if recommendation_manager is None:
        return JSONResponse(
            content={
                "error": "Recomendaciones IA no habilitadas",
                "hint": "Configura GROQ_API_KEY en tu archivo .env",
            },
            status_code=503,
        )
    try:
        db = DatabaseManager()

        peak_hours = db.get_algorithm_results("peak_hour")
        weather = db.get_algorithm_results("Weather prediction")
        predictions = db.get_algorithm_results("Prediction")
        total = db.get_total_entries()

        db.close()

        datos_prediccion = {
            "hora_pico": peak_hours[0] if peak_hours else None,
            "prediccion_clima": weather[0] if weather else None,
            "prediccion_futuro": predictions[0] if predictions else None,
            "total_entradas": total,
        }

        resultado = recommendation_manager.generate(datos_prediccion)

        if resultado["status"] == "success":
            try:
                db = DatabaseManager()
                connection = db._get_connection()
                cursor = connection.cursor()

                query = """
                    INSERT INTO recomendaciones (algoritmo, timestamp, resultado)
                    VALUES ('AI_Recommendation', %s, %s)
                """

                cursor.execute(
                    query, (datetime.now(), json.dumps(resultado, ensure_ascii=False, indent=2))
                )

                cursor.close()
                connection.close()
                db.close()

                print("Recomendación guardada en BD")
            except Exception as e:
                print(f"No se pudo guardar en BD: {e}")

        return resultado

    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "status": "error"}, status_code=500
        )


@app.get("/api/recommendations/latest")
async def get_latest_ai_recommendation():

    try:
        db = DatabaseManager()
        connection = db._get_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT * FROM recomendaciones 
            WHERE algoritmo = 'AI_Recommendation'
            ORDER BY timestamp DESC 
            LIMIT 1
        """
        cursor.execute(query)
        result = cursor.fetchone()

        cursor.close()
        connection.close()
        db.close()

        if result:
            if result["timestamp"]:
                result["timestamp"] = result["timestamp"].isoformat()

            if result["resultado"]:
                result["resultado"] = json.loads(result["resultado"])

            print(f'Recomendacion ${result}')
            return result
        else:
            return {
                "message": "No hay recomendaciones previas",
                "hint": "Usa POST /api/recommendations/generate para crear una",
            }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT, reload=False)
