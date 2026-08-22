from abc import ABC, abstractmethod
from typing import List

import numpy as np

from utils.types import Detection


class PoseDetector(ABC):
    """Interface every detector backend implements.

    Swapping backends later (e.g. a TensorRT engine for the Jetson Nano) only
    means writing a new class here - main.py and the tracker never change.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        ...
