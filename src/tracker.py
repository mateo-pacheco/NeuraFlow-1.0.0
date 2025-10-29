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

    def add_position(self, x: int, y: int, height: int, width: int):
        now = time.time()
        self.positions.append((x, y, now, height, width))
        self.last_seen = now

        if len(self.positions) > 15:
            self.positions = self.positions[-15:]

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


class PersonTracker:

    def __init__(self):
        self.tracked_people: Dict[int, TrackedPerson] = {}
        self.next_person_id: int = 1
        self.distance_threshold = settings.DISTANCE_TRACKING
        self.timeout = settings.TRACKING_TIMEOUT

    def update(self, detections: List[Tuple]) -> Dict[int, TrackedPerson]:

        self._cleanup_old_tracks()

        assigned_ids = set()

        for detection in detections:
            x1, y1, x2, y2, conf = detection

            center_x = (x1 + x2) // 2
            bottom_y = y2

            bbox_width = x2 - x1
            bbox_height = y2 - y1

            person_id = self._find_closest_person(center_x, bottom_y, assigned_ids)

            if person_id is not None:
                self.tracked_people[person_id].add_position(
                    center_x, bottom_y, bbox_height, bbox_width
                )
                assigned_ids.add(person_id)
            else:
                person_id = self._create_new_person(
                    center_x, bottom_y, bbox_height, bbox_width
                )
                assigned_ids.add(person_id)

        return self.tracked_people

    def _find_closest_person(self, x: int, y: int, assigned_ids: set) -> Optional[int]:

        closest_person_id = None
        min_distance = float("inf")

        for person_id, person in self.tracked_people.items():
            if person_id in assigned_ids:
                continue

            last_position = person.get_last_position()

            if last_position is None:
                continue

            last_x, last_y = last_position

            distance = np.hypot(x - last_x, y - last_y)

            if distance < self.distance_threshold and distance < min_distance:
                min_distance = distance
                closest_person_id = person_id

        return closest_person_id

    def _create_new_person(self, x: int, y: int, height: int, width: int) -> int:
        person_id = self.next_person_id
        self.next_person_id += 1

        person = TrackedPerson(person_id=person_id)
        person.add_position(x, y, height, width)

        self.tracked_people[person_id] = person

        return person_id

    def _cleanup_old_tracks(self):
        to_remove = []

        for person_id, person in self.tracked_people.items():
            if person.time_since_last_seen() > self.timeout:
                to_remove.append(person_id)

        for person_id in to_remove:
            del self.tracked_people[person_id]

        if to_remove:
            print(f"Limpieza: {len(to_remove)} persona(s) eliminada(s)")

    def get_person(self, person_id: int) -> Optional[TrackedPerson]:
        return self.tracked_people.get(person_id)

    def get_all_people(self) -> Dict[int, TrackedPerson]:
        return self.tracked_people

    def count_active_tracks(self) -> int:
        return len(self.tracked_people)

    def reset(self):
        self.tracked_people = {}
        self.next_person_id = 1
        print("Tracker resetado")

    def mark_as_counted(self, person_id: int):

        if person_id in self.tracked_people:
            self.tracked_people[person_id].counted = True
