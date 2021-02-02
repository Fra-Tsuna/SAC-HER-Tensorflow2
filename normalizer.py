import numpy as np

class Normalizer:
    def __init__(self, size, eps=1e-2, clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.clip_range = clip_range

        # some local information
        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)

        # get the total sum sumsq and sum count
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)

        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)
    
    def update(self, v):
        v = v.reshape(-1, self.size)
        self.local_sum += v.sum(axis=0)
        self.local_sumsq += (np.square(v)).sum(axis=0)
        self.local_count[0] += v.shape[0]

    def recompute_stats(self):
        self.mean = self.local_sum / self.local_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.local_sumsq / self.local_count) - np.square(self.local_sum / self.local_count)))

    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.clip_range
        return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)
