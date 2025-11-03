import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


class Settings:
    
    # Proyecto
    PROJECT_NAME = "NeuraFlow"
    VERSION = "1.0.0"  # Versión actualizada
    
    # Base de datos
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", "3306"))
    DB_NAME = os.getenv("DB_NAME", "neuraflow")
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "admin")
    DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
    
    # Cámara
    CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
    CAMERA_RECONNECT_RETRIES = int(os.getenv("CAMERA_RECONNECT_RETRIES", "3"))
    CAMERA_RECONNECT_DELAY = float(os.getenv("CAMERA_RECONNECT_DELAY", "1.0"))
    
    # Detección YOLO
    MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
    MODEL_VERSION = "YOLOv8n"
    
    # ACTUALIZADO: Umbrales de confianza más altos
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))
    MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.4"))
    
    # Validación de bbox
    MIN_HEIGHT = 70
    MIN_AREA_RATIO = 0.0015
    MAX_AREA_RATIO = 0.35
    MIN_ASPECT_RATIO = 1.3
    MAX_ASPECT_RATIO = 4.0

    # ACTUALIZADO: Tracking mejorado
    TRACKING_TIMEOUT = float(os.getenv("TRACKING_TIMEOUT", "1.5"))
    DISTANCE_TRACKING = int(os.getenv("DISTANCE_TRACKING", "150"))
    MAX_FRAMES_LOST = int(os.getenv("MAX_FRAMES_LOST", "10"))
    
    # Aproximación
    DIRECTION_THRESHOLD = int(os.getenv("DIRECTION_THRESHOLD", "30"))
    FRAMES_MIN_DETECTION = int(os.getenv("FRAMES_MIN_DETECTION", "8"))
    RATIO_APPROACH = float(os.getenv("RATIO_APPROACH", "0.15"))
    
    # API
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    # NUEVO: Optimización de performance
    PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", "2"))
    
    # Performance
    BATCH_DB_INSERTS = os.getenv("BATCH_DB_INSERTS", "true").lower() == "true"
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
    JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))
    FPS_UPDATE_INTERVAL = int(os.getenv("FPS_UPDATE_INTERVAL", "30"))
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    CONFIG_DIR = BASE_DIR / "config"
    LOGS_DIR = BASE_DIR / "logs"
    MODELS_DIR = BASE_DIR / "models"
    
    # Crear directorios si no existen
    LOGS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_line_config_path(cls) -> Path:
        return cls.CONFIG_DIR / "line_config.json"
    
    @classmethod
    def validate(cls):
        assert 0.0 <= cls.CONFIDENCE_THRESHOLD <= 1.0, "CONFIDENCE_THRESHOLD debe estar entre 0 y 1"
        assert cls.MIN_CONFIDENCE >= cls.CONFIDENCE_THRESHOLD, "MIN_CONFIDENCE debe ser >= CONFIDENCE_THRESHOLD"
        assert cls.MIN_AREA_RATIO < cls.MAX_AREA_RATIO, "MIN_AREA_RATIO debe ser < MAX_AREA_RATIO"
        assert cls.BATCH_SIZE > 0, "BATCH_SIZE debe ser > 0"
        assert cls.PROCESS_EVERY_N_FRAMES >= 1, "PROCESS_EVERY_N_FRAMES debe ser >= 1"
        assert cls.MAX_FRAMES_LOST > 0, "MAX_FRAMES_LOST debe ser > 0"


# Validar al importar
Settings.validate()

# Exportar instancia
settings = Settings()