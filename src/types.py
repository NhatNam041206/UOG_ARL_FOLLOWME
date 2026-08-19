from dataclasses import dataclass
from typing import Optional


@dataclass
class AngleResult:
    target_found: bool
    angle_offset_deg: Optional[float] = None
    size_ratio: Optional[float] = None
    track_id: Optional[int] = None
    similarity_score: Optional[float] = None
