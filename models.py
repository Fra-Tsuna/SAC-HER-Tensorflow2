import numpy as np
import tensorflow as tf
import keras
from keras import *
from keras.layers import *
from tensorflow.keras.optimizers import Adam

class CriticNetwork():
    def __init__(self, beta, input_dims, num_actions, l1_dims=256, l2_dims=256):
        self.input_dims=input_dims
        self.l1_dims=l1_dims
        self.l2_dims=l2_dims
        self.num_actions=num_actions

        self.model=Sequential()
        self.model.add(Input(self.input_dims[0]+num_actions,))
        self.model.add(Dense(self.l1_dims))
        self.model.add(ReLU)
        self.model.add(Dense(self.l2_dims))
        self.model.add(ReLU)
        self.model.add(Dense(1))
        self.optimizer=Adam(learning_rate=beta)
        self.model.compile(optimizer=optimizer)
    
    def forward(self, state, action):
        return self.model.predict(concatenate(state, action))
