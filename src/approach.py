import numpy as np
from typing import Tuple, List

from config.settings import settings
from src.tracker import TrackedPerson

def is_approaching_camera(person: TrackedPerson) -> Tuple[bool, str]:

    position = person.positions

    min_frames = settings.FRAMES_MIN_DETECTION

    if len(position) < min_frames:
        return False, "Frames insuficientes"

    recent_positions = position[-min_frames:]

    area = [pos[3] * pos[4] for pos in recent_positions]
    y_positions = [pos[1] for pos in recent_positions]

    # Crecimiento del area del bounding box
    initial_area = np.mean(area[:2])
    final_area = np.mean(area[-2:])
    
    if initial_area > 0:
        area_growth_ratio = (final_area - initial_area) / initial_area
    else:
        area_growth_ratio = 0
    
    bbox_growing = area_growth_ratio > settings.RATIO_APPROACH

    # Movimiento vertical hacia abajo
    y_trend = np.polyfit(range(len(y_positions)), y_positions, 1)[0]

    min_y_movement_per_frame = settings.DIRECTION_THRESHOLD / min_frames
    
    moving_down = y_trend > min_y_movement_per_frame

    # Consistencia de movimiento
    y_changes = [
        y_positions[i] - y_positions[i - 1]
        for i in range(1, len(y_positions))
    ]

    frames_moving_down = sum(1 for change in y_changes if change > 5)

    if y_changes:
        consistency_ratio = frames_moving_down / len(y_changes)
    else:
        consistency_ratio = 0

    is_moving = consistency_ratio >= 0.7

    # Decision final

    is_approaching = bbox_growing and moving_down and is_moving

    debug_info = (
        f"Crecimiento del area: {area_growth_ratio:.2f}\n"
        f"Movimiento vertical: {y_trend:.2f}\n"
        f"Consistencia de movimiento: {consistency_ratio:.2f}\n"
    )

    return is_approaching, debug_info

def check_line_crossing(person: TrackedPerson, line: List[int]) -> bool:

    if not person.positions:
        return False

    _, bottom_y, _, bbox_height, _ = person.positions[-1]

    top_y = bottom_y - bbox_height

    line_y = (line[1] + line[3]) // 2

    crossed_line = (top_y < line_y) and (bottom_y >= line_y)

    return crossed_line

def validate_entry(person: TrackedPerson, line: List[int]) -> Tuple[bool, str]:
    
    # Verifica si ya fue contado
    if person.counted:
        return False, "Ya fue contado"
    
    # Si cruzo la linea
    crossed_line = check_line_crossing(person, line)
    
    if not crossed_line:
        return False, "No cruzo la linea"
    
    # Si se acerca a la camara
    
    is_approaching, debug_info = is_approaching_camera(person)
    
    if not is_approaching:
        return False, "No se acerca a la camara"
    
    return True, f"Entada valida ({debug_info})"

def get_approach_score(person: TrackedPerson) -> float:

    position = person.positions
    min_frames = settings.FRAMES_MIN_DETECTION

    if len(position) < min_frames:
        return 0.0
    
    recent_positions = position[-min_frames:]

    area = [pos[3] * pos[4] for pos in recent_positions]
    initial_area = np.mean(area[:2])
    final_area = np.mean(area[-2:])
    
    if initial_area > 0:
        area_ratio = (final_area - initial_area) / initial_area
        score = min(1.0, max(0.0, area_ratio / settings.RATIO_APPROACH))
    else:
        area_ratio = 0.0
    
    
    return score
