from abc import ABC, abstractmethod
from typing import List

from utils.types import Detection, TrackedObject


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        ...
