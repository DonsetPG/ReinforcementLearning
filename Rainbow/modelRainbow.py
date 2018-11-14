import math
import tensorflow as tf 
import numpy as np 
import keras 
from keras import backend as K

#############

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Conv2D,AveragePooling2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, GaussianNoise

#############

class DQN():

    def __init__(self,args,action_size,state_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.lr = args.lr
        self.history_length = args.history_length
        self.noise = args.noise
        self.noisy_std = args.noisy_std
        self.atoms = args.atoms

    def load(self,path):
        new_model = load_model(path)
        self.model = new_model 
        self.update_target_model()

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = keras.Sequential()
        model.add(Conv2D(16,(8,8),stride=4, activation='relu', 
                         data_format="channels_first", input_shape=(self.history_length,self.state_size,self.state_size)) )
        model.add(Conv2D(32,(4,4),stride=2, activation='relu', 
                         data_format="channels_first"))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        if self.noise:
            model.add(GaussianNoise(stddev=self.noisy_std))
        model.add(Dense(self.action_size*self.atoms, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None))
        model.summary()
        return model

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def predict(self,state):
        return self.model.predict(state).reshape(self.atoms,self.action_size)

        