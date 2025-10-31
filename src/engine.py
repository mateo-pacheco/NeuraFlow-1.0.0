import cv2
import time
from datetime import datetime
from typing import Optional, Callable
import numpy as np
from typing import List

from config.settings import settings
from src.camera import CameraManager
from src.detector import PersonDetector
from src.tracker import PersonTracker
from src.approach import validate_entry, check_line_crossing, is_approaching_camera
from src.database import DatabaseManager, Entry, create_database
from src.utils import FPSCalculator, load_json_config, print_header, print_info

class DetectionEngine:

    def __init__(self, use_database: bool = True):
        print_header("NeuraFlow - Sistema de Detección de Entradas con IA")
        self.camera = CameraManager()
        self.detector = PersonDetector()
        self.tracker = PersonTracker()
        self.db_manager = None

        if use_database:
            try:
                create_database()
                self.db_manager = DatabaseManager()
            except Exception as e:
                print(f"Base de datos deshabilitada: {e}")
        
        self.total_entries = 0
        self.frame_count = 0
        self.fps_calculator = FPSCalculator(settings.FPS_UPDATE_INTERVAL)
        self.is_running = False

        self.process_every_n_frames = settings.PROCESS_EVERY_N_FRAMES
        self.last_detections = []

        self.line = self._load_line_config()

        print_info("Camara", self.camera.source)
        print_info("Modelo", settings.MODEL_PATH)
        print_info("Base de datos", "Activa" if self.db_manager else "Desactivada")
        print_info("Linea", "Configurada" if self.line else "Por defecto")
        print_info("Optimización", f"Procesa 1/{self.process_every_n_frames} frames")
        print("=" * 70)
    
    def _load_line_config(self) -> Optional[List]:
        config_path = settings.get_line_config_path()
        config = load_json_config(config_path)
        return config.get("line")
    
    def start(self, frame_callback: Optional[Callable] = None, show_window: bool = True):

        if not self.camera.open():
            print("Error: No se pudo abrir la camara")
            return
        
        ret, frame = self.camera.read()
        if not ret:
            print("Error: No se pudo leer el frame")
            return
        
        height, width = frame.shape[:2]

        if self.line is None:
            self.line = [0, height // 2, width, height // 2]
            print_info("Linea", f"Por defecto - Y = {height // 2}")

        self.is_running = True
        print("Motor iniciado")
        print("Presiona 'Q' para salir, 'R' para reiniciar, '+/-' para ajustar velocidad")
        print("=" * 70)
        
        try:
            while self.is_running:
                ret, frame = self.camera.read()
                if not ret:
                    print("Error: No se pudo leer el frame")
                    time.sleep(0.1)
                    continue

                should_detect = (self.frame_count % self.process_every_n_frames) == 0
                
                frame = self._process_frame(frame, detect=should_detect)

                self.frame_count += 1
                fps = self.fps_calculator.update(self.frame_count)
                
                frame = self._add_ui_overlay(frame, fps)

                if frame_callback:
                    frame_callback(frame)
                
                show_window = True
                if show_window:
                    cv2.imshow("NeuraFlow", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Saliendo")
                        break
                    elif key == ord('r'):
                        self.reset_counter()
                    elif key == ord('+'):
                        self.process_every_n_frames = max(1, self.process_every_n_frames - 1)
                        print(f"Velocidad: 1/{self.process_every_n_frames} frames")
                    elif key == ord('-'):
                        self.process_every_n_frames = min(5, self.process_every_n_frames + 1)
                        print(f"Velocidad: 1/{self.process_every_n_frames} frames")
        finally:
            self.stop()
    
    def _process_frame(self, frame: np.ndarray, detect: bool = True) -> np.ndarray:
        height, width = frame.shape[:2]

        x1, y1, x2, y2 = self.line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 128), 2)

        if detect:
            detections = self.detector.detect(frame)
            self.last_detections = detections
        else:
            detections = self.last_detections

        self.tracker.update(detections)

        active_people = self.tracker.get_active_people()

        for person_id, person in active_people.items():
            last_position = person.get_last_position()
            if last_position is None:
                continue

            center_x, bottom_y = last_position

            if not person.positions:
                continue

            _, _, _, bbox_height, bbox_width = person.positions[-1]
            x1 = center_x - bbox_width // 2
            y1 = bottom_y - bbox_height
            x2 = center_x + bbox_width // 2
            y2 = bottom_y

            is_valid, reason = validate_entry(person, self.line)

            if is_valid and not person.counted:
                self._register_entry(person_id, center_x, bottom_y)
            
            crossed_line = check_line_crossing(person, self.line)
            is_approaching, _ = is_approaching_camera(person)

            color = self._get_bbox_color(crossed_line, is_approaching, person.counted)

            thickness = 3 if person.counted else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            label = f"ID:{person_id}"
            if person.counted:
                label += " [CONTADO]"
            
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame, 
                (x1, y1 - text_height - 8), 
                (x1 + text_width + 4, y1), 
                color, 
                -1
            )
            
            cv2.putText(
                frame, label, 
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 1
            )

            cv2.circle(frame, (center_x, bottom_y), 5, (146, 22, 168), -1)

        return frame

    def _register_entry(self, person_id: int, x: int, y: int):
        self.total_entries += 1
        self.tracker.mark_as_counted(person_id)

        timestamp = datetime.now()

        if self.db_manager:
            person = self.tracker.get_person(person_id)
            entry = Entry(
                timestamp=timestamp,
                total_entries=self.total_entries,
                x_center=x,
                y_bottom=y,
                confidence=person.confidence,
                model_version=settings.MODEL_VERSION
            )
            self.db_manager.insert_entry(entry)
        
        print(f"[{timestamp:%H:%M:%S}] ✓ Entrada #{self.total_entries} - ID:{person_id}")

    def _get_bbox_color(self, crossed_line: bool, is_approaching: bool, 
                       counted: bool) -> tuple:
        if counted:
            return (128, 128, 128)
        elif crossed_line and is_approaching:
            return (34, 171, 116)
        elif crossed_line:
            return (0, 165, 255)
        elif is_approaching:
            return (255, 0, 0)
        else:
            return (255, 255, 0)
    
    def _add_ui_overlay(self, frame: np.ndarray, fps: float) -> np.ndarray:
        cv2.putText(
            frame,
            f"Entradas: {self.total_entries}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3,
            lineType=cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"Entradas: {self.total_entries}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA
        )

        active = self.tracker.count_active_tracks()
        info_text = f"FPS: {fps:.1f} | Tracking: {active} | Skip: 1/{self.process_every_n_frames}"
        cv2.putText(
            frame,
            info_text,
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA
        )

        return frame
    
    def reset_counter(self):
        self.total_entries = 0
        self.tracker.reset()
        print("✓ Contador reiniciado")
    
    def stop(self):
        self.is_running = False

        if self.db_manager:
            self.db_manager.force_flush()
            self.db_manager.close()
        
        self.camera.release()
        cv2.destroyAllWindows()
        
        print_header("SESIÓN FINALIZADA")
        print_info("Total entradas", self.total_entries)
        print_info("Frames procesados", self.frame_count)
        print("=" * 70)
    
    def get_statistics(self) -> dict:
        stats = {
            'total_entries': self.total_entries,
            'fps': self.fps_calculator.fps,
            'tracked_people': self.tracker.count_active_tracks(),
            'frame_count': self.frame_count,
            'db_connected': self.db_manager is not None,
            'process_rate': f"1/{self.process_every_n_frames}"
        }

        if self.db_manager:
            db_stats = self.db_manager.get_statistics()
            stats['db_total_entries'] = db_stats.total_entries
            stats['db_avg_confidence'] = db_stats.prom_confidence
            stats['daily_entries'] = db_stats.daily_entry
        
        return stats
