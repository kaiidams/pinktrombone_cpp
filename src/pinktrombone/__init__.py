from ._pinktrombone import _PinkTrombone
import numpy as np


class PinkTrombone:
    def __init__(self, n):
        self._pinktrombone = _PinkTrombone(n)

    def control(self, data):
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float64)
        self._pinktrombone.control(data.tobytes())

    def process(self):
        data = self._pinktrombone.process()
        return np.frombuffer(data, dtype=np.float64)
