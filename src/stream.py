import cv2
import threading
from typing import Optional
import numpy as np
import traceback
from config.settings import settings

from src.engine import DetectionEngine

class StreamHandler:
    def __init__(self, use_database: bool = True):
        
        self.engine = DetectionEngine(use_database)

        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()

    def start(self):
        if self.is_running:
            print("StreamHandler ya esta corriendo")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._run_detection, daemon=True)
        self.thread.start()
        print("StreamHandler iniciado")
        
    def stop(self):
        if not self.is_running:
            return
        
        print("Deteniendo StreamHandler")
        self.is_running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)

        self.engine.stop()
        print("StreamHandler detenido")
    
    def _run_detection(self):
        try:
            self.engine.start(
                frame_callback = self._save_frame,
                show_window = False
            )
        except Exception as e:
            print(f"Error en StreamHandler: {e}")
            traceback.print_exc()
            self.is_running = False
    
    def _save_frame(self, frame: np.ndarray):
        with self.frame_lock:
            self.current_frame = frame.copy()
    
    def get_frame(self) -> Optional[np.ndarray]:
        with self.frame_lock:
            if self.current_frame is None:
                return None
            return self.current_frame.copy()
    
    def get_jpeg_frame(self, quality: int = None) -> Optional[bytes]:
        
        frame = self.get_frame()

        if frame is None:
            return None
        
        if quality is None:
            quality = settings.JPEG_QUALITY
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

        ret, buffer = cv2.imencode('.jpg', frame, encode_param)

        if not ret:
            return None
        
        return buffer.tobytes()
    
    def get_statistics(self) -> dict:
        return self.engine.get_statistics()

    def reset_counter(self):
        self.engine.reset_counter()

    def is_alive(self) -> bool:
        return self.is_running and self.thread and self.thread.is_alive()
        
        