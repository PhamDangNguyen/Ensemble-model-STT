# lru_cache is used to avoid redundant computation
from functools import lru_cache

class Segment:

    def __init__(self, word, start, end,confidence,model_stride=8, window_stride=0.01):
        self.word = word
        self._start = start
        self._end = end
        self.model_stride = model_stride
        self.confidence = confidence
        self.window_stride = window_stride

    def __repr__(self):
        return f"Segment({self.word}, start:{self.start:.2f}, end:{self.end:.2f}, confidence:{self.confidence:.2f})"

    @property
    @lru_cache
    def start(self):
        return self._start*self.model_stride*self.window_stride
    
    @property
    @lru_cache
    def end(self):
        return self._end*self.model_stride*self.window_stride

    @property
    @lru_cache
    def duration(self):
        return self.end - self.start