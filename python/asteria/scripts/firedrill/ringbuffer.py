import numpy as np
from abc import ABC, abstractmethod


class HitBuffer(ABC):

    def __init__(self, size=80):
        self._size = size
        # self._nbins = nbins
        self._dtypes = [('hit_length', '>u4'),
                        ('hit_type', '>u4'),
                        ('dom_id', '>u8'),
                        ('unused', '>u8'),
                        ('utc_time', '>u8'),
                        ('byte_order', '>u2'),
                        ('version', '>u2'),
                        ('pedestal', '>u2'),
                        ('dom_clk', '>u8'),
                        ('word1', '>u4'),
                        ('word3', '>u4')]
        super().__init__()

    @abstractmethod
    def append(self, entry):
        ...

    @abstractmethod
    def clear(self):
        ...

    def __getitem__(self, key):
        return self.data[key]

    @property
    @abstractmethod
    def data(self):
        ...


class WindowBuffer(HitBuffer):
    def __init__(self, size, mult=2):
        super().__init__(size)
        self._mult = mult
        self._buflen = self._size*self._mult
        self.clear()

    def append(self, entry):
        if self._idx >= self._buflen:
            self._reset()
        self._data[self._idx] = entry
        self._idx += 1
        return self

    def clear(self):
        self._data = np.zeros(self._buflen, dtype=self._dtypes)
        self._idx = self._size

    def _reset(self):
        self._idx = self._size
        self._data[:self._size] = self._data[-self._size:]

    def __getitem__(self, key):
        return self.data[key]

    @property
    def data(self):
        return self._data[self._idx-self._size:self._idx]
