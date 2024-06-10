from ._pinktrombone import _PinkTrombone
import numpy as np


class PinkTrombone:
    def __init__(self, n):
        self.n = n
        self.sample_rate = 44100
        self.block_size = 512
        self.reset()

    def reset(self):
        self._pinktrombone = _PinkTrombone(self.n)

    def control(self, data):
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float64)
        self._pinktrombone.control(data.tobytes())

    def process(self):
        data = self._pinktrombone.process()
        return np.frombuffer(data, dtype=np.float64)

    def __call__(self, data):
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float64)
        out = []
        for x in data:
            self.control(x)
            y = self.process()
            out.append(y)
        return np.concatenate(out)
