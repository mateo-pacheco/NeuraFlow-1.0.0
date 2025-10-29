import re
import cv2
import time
from typing import Optional, Tuple
from config.settings import settings
from src.utils import CameraError, retry


class CameraManager:

    def __init__(self, source: Optional[str] = None):
        self.source = self._resolve_source(source)
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        self.frame_count = 0

        print(f"Cámara inicializada con fuente: {self.source}")

    def _resolve_source(self, source: Optional[str]) -> str:
        if source is None:
            source = settings.CAMERA_SOURCE
        return str(source).strip()

    def open(self) -> bool:
        for attempt in range(settings.CAMERA_RECONNECT_RETRIES):
            if self._try_open():
                self.is_opened = True
                return True

            if attempt < settings.CAMERA_RECONNECT_RETRIES - 1:
                delay = settings.CAMERA_RECONNECT_DELAY
                print(f"Reintentando conexión en {delay} segundos...")
                time.sleep(delay)

        print("No se pudo establecer la conexión con la cámara")
        self.is_opened = False
        return False

    def _try_open(self) -> bool:

        if self.cap:
            self.cap.release()
            self.cap = None

        if self._is_network_stream():
            return self._open_network_camera()
        elif self.source.isdigit():
            return self._open_local_camera(int(self.source))
        else:
            print(f"Fuente no valida: {self.source}")
            return False

    def _is_network_stream(self) -> bool:
        return self.source.startswith(("rtsp://", "http://", "https://"))

    def _open_network_camera(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.source)

            if self.cap.isOpened():
                print(f"Cámara abierta con fuente: {self.source}")
                return True
            else:
                print(f"No se pudo abrir la cámara con fuente: {self.source}")
                return False
        except Exception as e:
            print(f"Error al abrir la cámara con fuente: {self.source}")
            return False

    def _open_local_camera(self, index: int) -> bool:
        backends = [
            (cv2.CAP_DSHOW, "DSHOW"),  # DirectShow (Windows)
            (cv2.CAP_MSMF, "MSMF"),  # Media Foundation (Windows)
            (cv2.CAP_V4L2, "V4L2"),  # Video4Linux (Linux)
            (cv2.CAP_ANY, "ANY"),  # Backend automático
        ]

        for backend_id, backend_name in backends:
            try:
                self.cap = cv2.VideoCapture(index, backend_id)

                if self.cap.isOpened():
                    print(
                        f"Cámara abierta con fuente: {self.source} usando backend: {backend_name}"
                    )
                    return True
            except Exception as e:
                print(f"Error al abrir la cámara con fuente: {self.source}")
                continue

        print(f"No se pudo abrir la cámara con fuente: {self.source}")
        return False

    def read(self) -> Tuple[bool, Optional[any]]:
        if not self.cap or not self.is_opened:
            print("Laa camara no esta abierta. Intentando reabrir...")
            if not self.open():
                print("No se pudo reabrir la cámara")
                return False, None
        ret, frame = self.cap.read()

        if not ret:
            print("Error al leer el frame")
            if self.reconnect():
                ret, frame = self.cap.read()

        if ret:
            self.frame_count += 1

        return ret, frame

    def reconnect(self) -> bool:
        print("Reconectando la cámara...")
        return self.open()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            self.is_opened = False
            print("Cámara liberada")

    def is_ready(self) -> bool:
        return self.cap is not None and self.is_opened

    def get_properties(self) -> dict:

        if not self.is_ready():
            return {}

        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(self.cap.get(cv2.CAP_PROP_FPS)),
            "frame_count": self.frame_count,
        }

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
