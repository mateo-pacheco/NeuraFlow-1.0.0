import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from functools import wraps

def load_json_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

def save_json_config(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

# Serializacion de los Json
def change_date(obj: Any) -> Any:
    if  isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def save_json_datetime(data: Any) -> Any:
    return json.dump(data, default=change_date, ensure_ascii=False)

# Mide el tiempo que tarda en ejecutarse un bloque de codigo
class Timer:
    def __init__(self, name: str = "Operacion"):
        self.name = name
        self.start = None
        self.elapsed = None
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(sel, *args):
        self.elapsed = time.time() - self.start
        print(f"{self.name}: {self.elapsed:.3f}s")

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, *kwargs)
        elapsed = time.time() - start
        print('{func.__name__}: {elapsed:.3f}s')
        return result
    return wrapper

# Calcular los FPS
class FPSCalculator:
    def __init__(self, update_inteval: int = 30):
        self.update_inteval = update_inteval
        self.last_time = time.time()
        self.last_frame_count = 0
        self._fps = 0

    def update(self, current_frame_count: int) -> float:
        if current_frame_count % self.update_inteval == 0:
            current_time = time.time()
            elapsed = current_time - self.last_time
            frames = current_frame_count - self.last_frame_count

            if elapsed > 0:
                self._fps = frames / elapsed
                self.last_time = current_time
                self.last_frame_count = current_frame_count

        return self._fps
    
    @property
    def fps(self) -> float:
        return self._fps

# Validar si el Bounding Box es valido
def validate_bbox(bbox: tuple, frame_shape: tuple, 
                  min_height: int = 60,
                  min_area_ratio: float = 0.001,
                  max_area_ratio: float = 0.4,
                  min_aspect_ratio: float = 1.2,
                  max_aspect_ratio: float = 4.5) -> bool:
                  
    x1, y1, x2, y2 = bbox
    frame_height, frame_width = frame_shape[:2]
    
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    if bbox_height < min_height:
        return False
    
    bbox_area = bbox_width * bbox_height
    frame_area = frame_width * frame_height
    area_ratio = bbox_area / frame_area
    
    if not (min_area_ratio < area_ratio < max_area_ratio):
        return False
    
    if bbox_width == 0:
        return False
    
    aspect_ratio = bbox_height / bbox_width
    if not (min_aspect_ratio < aspect_ratio < max_aspect_ratio):
        return False
    
    return True

# Obtener el centro del bbox (Pie de la persona)
def get_center_bbox(bbox: tuple) -> tuple:
    x1, x2, y1, y2 = bbox
    center_x = int((x1 + x2) // 2)
    center_y = int(y2)
    return center_x, center_y

# Formato de impresion en consola
# ============================================================================
# Formateo y Display
# ============================================================================

def format_timestamp(dt: Optional[datetime] = None) -> str:
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def print_header(title: str, width: int = 70):
    print("=" * width)
    print(title.center(width))
    print("=" * width)


def print_info(label: str, value: Any):
    print(f"  {label}: {value}")

# Funcion para reintentar la ejecucion de una funcion
def retry(max_attemps: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attemps):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt < max_attemps - 1:
                        raise
                    print(f"Error al ejecutar {func.__name__}: {e}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

# Excepciones
class CameraError(Exception):
    """Error relacionado con la cámara"""
    pass


class DetectionError(Exception):
    """Error en la detección"""
    pass


class DatabaseError(Exception):
    """Error en la base de datos"""
    pass


class ConfigurationError(Exception):
    """Error en la configuración"""
    pass
