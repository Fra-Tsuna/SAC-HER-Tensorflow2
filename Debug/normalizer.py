#!/usr/bin/env python3

import numpy as np
import time
import tensorflow as tf


DEBUG_NORMALIZE = False
DEBUG_UPDATE = False
NORMALIZATION = "Gaussian"


class Normalizer:
    def __init__(self, size, eps=1e-2, clip_range=np.inf, 
                 normalization=NORMALIZATION):
        self.size = size
        self.eps = eps
        self.clip_range = clip_range
        self.normalization = normalization

        if self.normalization == "Gaussian":
            self.local_sum = np.zeros(self.size, np.float32)
            self.local_sumsq = np.zeros(self.size, np.float32)
            self.local_count = np.zeros(1, np.float32)
            self.mean = np.zeros(self.size, np.float32)
            self.std = np.ones(self.size, np.float32)
        elif self.normalization == "MinMax":
            self.min = np.zeros(self.size, np.float32)
            self.max = np.ones(self.size, np.float32)
        else:
            raise TypeError("Wrong normalization type")

    def update(self, buffer):
        if self.normalization == "Gaussian":
            self.local_sum += buffer.sum(axis=0)
            self.local_sumsq += (np.square(buffer)).sum(axis=0)
            self.local_count[0] += buffer.shape[0]
            self.mean = self.local_sum / self.local_count
            self.std = np.sqrt(np.maximum(np.square(self.eps), 
                               (self.local_sumsq / self.local_count) - 
                                np.square(self.local_sum / self.local_count)))
        elif self.normalization == "MinMax":
            self.min = np.minimum(self.min, buffer.min(axis=0))
            self.max = np.maximum(self.max, buffer.max(axis=0))
        if DEBUG_UPDATE:
            print("++++++++++++++++ DEBUG - UPDATE NORMALIZER [NORMALIZER.UPDATE] ++++++++++++++++\n")
            if self.normalization == "Gaussian":
                print("----------------------------mean----------------------------")
                print(self.mean)
                print("----------------------------std-dev----------------------------")
                print(self.std)
                print("----------------------------local_sum----------------------------")
                print(self.local_sum)
                print("----------------------------local_count----------------------------")
                print(self.local_count)
                print("----------------------------local_sumsq----------------------------")
                print(self.local_sumsq)
            elif self.normalization == "MinMax":
                print("----------------------------mean----------------------------")
                print(self.min)
                print("----------------------------std-dev----------------------------")
                print(self.max)
            a = input("\n\nPress Enter to continue...")
        
    def normalize(self, vector, clip_range=None):
        if clip_range is None:
            clip_range = self.clip_range
        if self.normalization == "Gaussian":
            v_norm = (vector - self.mean) / self.std
        elif self.normalization == "MinMax":
            v_norm = (vector - self.min) / (self.max - self.min)
        if DEBUG_NORMALIZE:
            print("++++++++++++++++ DEBUG - NORMALIZE [NORMALIZER.NORMALIZE] ++++++++++++++++\n")
            if self.normalization == "Gaussian":
                print("----------------------------mean----------------------------")
                print(self.mean)
                print("----------------------------std-dev----------------------------")
                print(self.std)
                print("----------------------------local_sum----------------------------")
                print(self.local_sum)
                print("----------------------------local_count----------------------------")
                print(self.local_count)
                print("----------------------------local_sumsq----------------------------")
                print(self.local_sumsq)
            elif self.normalization == "MinMax":
                print("----------------------------mean----------------------------")
                print(self.min)
                print("----------------------------std-dev----------------------------")
                print(self.max)
            print("----------------------------vector----------------------------")
            print(v)
            print("----------------------------v_norm----------------------------")
            print(v_norm)
            a = input("\n\nPress Enter to continue...")
        return np.clip(v_norm, -clip_range, clip_range)
