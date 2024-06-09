from ._pinktrombone import _PinkTrombone
import numpy as np


class PinkTrombone:
    def __init__(self, n):
        self._pinktrombone = _PinkTrombone(n)

    def control(self, data):
        self._pinktrombone.control(data)

    def process(self):
        data = self._pinktrombone.process()
        return np.frombuffer(data, dtype=np.float64)
