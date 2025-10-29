import torch
from ultralytics import YOLO
from typing import List, Tuple
import numpy as np

from config.settings import settings
from src.utils import validate_bbox, DetectionError

class PersonDetector:

    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.device = self._detect_device()
        self.model = None

        self._load_model()

    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            print("Usando GPU")
            return "cuda"
        else:
            print("Usando CPU")
            return "cpu"
    
    def _load_model(self):
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print("Modelo cargado correctamente en: ", self.device)
        except Exception as e:
            raise DetectionError(f"Error al cargar el modelo: {e}")
    
    def detect(self, frame: np.ndarray) -> List[Tuple]:
        if self.model is None:
            raise DetectionError("Modelo no cargado")
        try:
            results = self.model(
                frame,
                conf = settings.CONFIDENCE_THRESHOLD,
                classes = [0],
                verbose = False,
            )

            return self._filter_detections(results, frame.shape)
        except Exception as e:
            print(f"Error al detectar personas: {e}")
            return []

    def _filter_detections(self, results, frame_shape) -> List[Tuple]:
        valid_detections = []
        
        if len(results) == 0 or results[0].boxes is None:
            return valid_detections
        
        boxes = results[0].boxes

        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())

            x1, y1, x2, y2 = map(int, box)
            bbox = (x1, y1, x2, y2)

            if self._is_valid_detection(bbox, conf, frame_shape):
                valid_detections.append((x1, y1, x2, y2, conf))
        
        return valid_detections

    def _is_valid_detection(self, bbox: Tuple, conf: float, frame_shape: Tuple) -> bool:
        if conf < settings.MIN_CONFIDENCE:
            return False
        return validate_bbox(
            bbox = bbox,
            frame_shape = frame_shape,
            min_height = settings.MIN_HEIGHT,
            min_area_ratio = settings.MIN_AREA_RATIO,
            max_area_ratio = settings.MAX_AREA_RATIO,
            min_aspect_ratio = settings.MIN_ASPECT_RATIO,
            max_aspect_ratio = settings.MAX_ASPECT_RATIO,
        )

    def get_bbox_info(self, bbox: Tuple) -> dict:
        x1, y1, x2, y2 = bbox

        width = x2 - x1
        height = y2 - y1
        area = width * height
        center_x =(x1 + x2) // 2
        center_y = (y1 + y2) // 2
        bottom_y = y2
        aspect_ratio = height / width if width > 0 else 0
        
        return {
            "width": width,
            "height": height,
            "area": area,
            "center_x": center_x,
            "center_y": center_y,
            "bottom_y": bottom_y,
            "aspect_ratio": aspect_ratio,
        }


        