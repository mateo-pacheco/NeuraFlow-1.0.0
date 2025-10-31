import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from config.settings import settings


@dataclass
class TrackedPerson:
    person_id: int
    positions: List[Tuple[int, int, float, int, int]] = field(default_factory=list)
    counted: bool = False
    last_seen: float = field(default_factory=time.time)
    confidence: float = 0.0
    frames_lost: int = 0
    total_detections: int = 0

    def add_position(self, x: int, y: int, height: int, width: int, confidence: float = 0.0):
        now = time.time()
        self.positions.append((x, y, now, height, width))
        self.last_seen = now
        self.confidence = confidence
        self.frames_lost = 0  # Reset al detectar
        self.total_detections += 1

        # Mantener historial más largo para mejor análisis
        if len(self.positions) > 20:
            self.positions = self.positions[-20:]

    def increment_frames_lost(self):
        """Incrementa contador cuando no se detecta en un frame"""
        self.frames_lost += 1

    def get_last_position(self) -> Optional[Tuple[int, int]]:
        if not self.positions:
            return None
        return self.positions[-1][:2]

    def get_position_history(self, frames: int = None) -> List[Tuple]:
        if not frames:
            return self.positions
        return self.positions[-frames:]

    def time_since_last_seen(self) -> float:
        return time.time() - self.last_seen
    
    def is_stable(self) -> bool:
        """Verifica si el tracking es estable (suficientes detecciones)"""
        return self.total_detections >= 3


class PersonTracker:

    def __init__(self):
        self.tracked_people: Dict[int, TrackedPerson] = {}
        self.next_person_id: int = 1
        self.distance_threshold = 150  # AUMENTADO de 80 a 150
        self.timeout = 1.5  # REDUCIDO de 5.0 a 1.5 segundos
        self.max_frames_lost = 10  # NUEVO: Máximo de frames sin detección

    def update(self, detections: List[Tuple]) -> Dict[int, TrackedPerson]:
        """
        Actualiza el tracking con las nuevas detecciones
        """
        # Incrementar frames perdidos para todos
        for person in self.tracked_people.values():
            person.increment_frames_lost()

        # Limpiar tracks antiguos ANTES de asignar
        self._cleanup_old_tracks()

        assigned_detection_indices = set()
        assigned_person_ids = set()

        # Ordenar detecciones por confianza (mayor primero)
        sorted_detections = sorted(enumerate(detections), 
                                   key=lambda x: x[1][4], reverse=True)

        # Primera pasada: asignar detecciones a personas existentes
        for det_idx, detection in sorted_detections:
            x1, y1, x2, y2, conf = detection

            center_x = (x1 + x2) // 2
            bottom_y = y2
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            person_id = self._find_closest_person(
                center_x, bottom_y, assigned_person_ids, conf
            )

            if person_id is not None:
                self.tracked_people[person_id].add_position(
                    center_x, bottom_y, bbox_height, bbox_width, conf
                )
                assigned_detection_indices.add(det_idx)
                assigned_person_ids.add(person_id)

        # Segunda pasada: crear nuevas personas para detecciones no asignadas
        for det_idx, detection in sorted_detections:
            if det_idx in assigned_detection_indices:
                continue

            x1, y1, x2, y2, conf = detection
            center_x = (x1 + x2) // 2
            bottom_y = y2
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            person_id = self._create_new_person(
                center_x, bottom_y, bbox_height, bbox_width, conf
            )
            assigned_person_ids.add(person_id)

        return self.tracked_people

    def _find_closest_person(self, x: int, y: int, assigned_ids: set, 
                            confidence: float) -> Optional[int]:
        """
        Encuentra la persona más cercana considerando distancia y confianza
        """
        closest_person_id = None
        min_score = float("inf")

        for person_id, person in self.tracked_people.items():
            if person_id in assigned_ids:
                continue

            last_position = person.get_last_position()
            if last_position is None:
                continue

            last_x, last_y = last_position
            distance = np.hypot(x - last_x, y - last_y)

            # Solo considerar si está dentro del threshold
            if distance > self.distance_threshold:
                continue

            # Score combinado: distancia y diferencia de confianza
            conf_diff = abs(confidence - person.confidence)
            score = distance + (conf_diff * 50)  # Penalizar cambios de confianza

            if score < min_score:
                min_score = score
                closest_person_id = person_id

        return closest_person_id

    def _create_new_person(self, x: int, y: int, height: int, width: int, 
                          confidence: float) -> int:
        """
        Crea una nueva persona tracked
        """
        person_id = self.next_person_id
        self.next_person_id += 1

        person = TrackedPerson(person_id=person_id)
        person.add_position(x, y, height, width, confidence)

        self.tracked_people[person_id] = person

        return person_id

    def _cleanup_old_tracks(self):
        """
        Elimina tracks antiguos o con muchos frames perdidos
        """
        to_remove = []

        for person_id, person in self.tracked_people.items():
            # Eliminar por timeout
            if person.time_since_last_seen() > self.timeout:
                to_remove.append(person_id)
                continue
            
            # NUEVO: Eliminar por frames perdidos consecutivos
            if person.frames_lost > self.max_frames_lost:
                to_remove.append(person_id)
                continue

        for person_id in to_remove:
            del self.tracked_people[person_id]

        if to_remove:
            print(f"Limpieza: {len(to_remove)} persona(s) eliminada(s)")

    def get_active_people(self) -> Dict[int, TrackedPerson]:
        """
        Retorna solo las personas activamente detectadas (no perdidas)
        """
        return {
            pid: person for pid, person in self.tracked_people.items()
            if person.frames_lost == 0 and person.is_stable()
        }

    def get_person(self, person_id: int) -> Optional[TrackedPerson]:
        return self.tracked_people.get(person_id)

    def get_all_people(self) -> Dict[int, TrackedPerson]:
        return self.tracked_people

    def count_active_tracks(self) -> int:
        return len(self.get_active_people())

    def reset(self):
        self.tracked_people = {}
        self.next_person_id = 1
        print("Tracker reseteado")

    def mark_as_counted(self, person_id: int):
        if person_id in self.tracked_people:
            self.tracked_people[person_id].counted = True