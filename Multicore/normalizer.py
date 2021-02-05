#!/usr/bin/env python3

import threading
import numpy as np
from mpi4py import MPI
import time
import tensorflow as tf


NORMALIZATION = "Gaussian"


class Normalizer:
    def __init__(self, size, eps=1e-2, clip_range=np.inf, normalization=NORMALIZATION):
        self.size = size
        self.eps = eps
        self.clip_range = clip_range
        self.normalization = normalization

        if self.normalization == "Gaussian":
            self.local_sum = np.zeros(self.size, np.float32)
            self.local_sumsq = np.zeros(self.size, np.float32)
            self.local_count = np.zeros(1, np.float32)
            self.total_sum = np.zeros(self.size, np.float32)
            self.total_sumsq = np.zeros(self.size, np.float32)
            self.total_count = np.ones(1, np.float32)
            self.mean = np.zeros(self.size, np.float32)
            self.std = np.ones(self.size, np.float32)
        elif self.normalization == "MinMax":
            self.min = np.zeros(self.size, np.float32)
            self.max = np.ones(self.size, np.float32)
        else:
            raise TypeError("Wrong normalization type")
        self.lock = threading.Lock()

    def update(self, buffer):
        with self.lock:
            if self.normalization == "Gaussian":
                self.local_sum += buffer.sum(axis=0)
                self.local_sumsq += (np.square(buffer)).sum(axis=0)
                self.local_count[0] += buffer.shape[0]
            elif self.normalization == "MinMax":
                self.min = np.minimum(self.min, buffer.min(axis=0))
                self.max = np.maximum(self.max, buffer.max(axis=0))

    # average across the cpu's data
    def _mpi_average(self, x):
        buf = np.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
        buf /= MPI.COMM_WORLD.Get_size()
        return buf
    
    # sync the parameters across the cpus
    def sync(self, local_sum, local_sumsq, local_count):
        local_sum[...] = self._mpi_average(local_sum)
        local_sumsq[...] = self._mpi_average(local_sumsq)
        local_count[...] = self._mpi_average(local_count)
        return local_sum, local_sumsq, local_count
    
    def recompute_stats(self):
        if self.normalization == "Gaussian":
            with self.lock:
                local_count = self.local_count.copy()
                local_sum = self.local_sum.copy()
                local_sumsq = self.local_sumsq.copy()
                # reset
                self.local_count[...] = 0
                self.local_sum[...] = 0
                self.local_sumsq[...] = 0
            # synrc the stats
            sync_sum, sync_sumsq, sync_count = self.sync(local_sum, local_sumsq, local_count)
            # update the total stuff
            self.total_sum += sync_sum
            self.total_sumsq += sync_sumsq
            self.total_count += sync_count
            # calculate the new mean and std
            self.mean = self.total_sum / self.total_count
            self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(self.total_sum / self.total_count)))
        
    def normalize(self, vector, clip_range=None):
        if clip_range is None:
            clip_range = self.clip_range
        if self.normalization == "Gaussian":
            v_norm = (vector - self.mean) / self.std
        elif self.normalization == "MinMax":
            v_norm = (vector - self.min) / (self.max - self.min)
        return np.clip(v_norm, -clip_range, clip_range)
