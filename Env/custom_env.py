from scipy.misc import imresize
import gym
import numpy as np
import random
import queue
from threading import Thread, RLock
from csb import Point,Unit,Pod,Collision,Game_only_runner,make_only_runner

## TODO :
# - add continuous (Mujoco for exemple)

verrou = RLock()
verrou1 = RLock()
verrou2 = RLock()




class CustomGym:
    def __init__(self, game_name, skip_actions=4, num_frames=4, w=100, h=100):
        with verrou:
            self.atari = True
            if game_name == 'Csb-only-runner-v0':
                self.env = make_only_runner()
                self.atari = False
            else:
                self.env = gym.make(game_name)
            self.num_frames = num_frames
            self.skip_actions = skip_actions
            self.w = w
            self.h = h
            self.opponent = False
            if game_name == 'SpaceInvaders-v0':
                self.action_space = [1,2,3] # For space invaders
            elif game_name == 'Phoenix-v0':
                self.action_space = [0,1,2,3,4]
            elif game_name == 'Boxing-v0':
                self.action_space = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
            elif game_name == 'Skiing-v0':
                self.action_space = [0,1,2]
            elif game_name == 'Tennis-v0':
                self.action_space = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
            elif game_name == 'Csb-only-runner-v0':
                self.action_space = [[-18,0],[-18,50],[-18,100],[-18,150],[-18,200],[-15,0],[-15,50],[-15,100],[-15,150],[-15,200],[-10,0],[-10,50],[-10,100],[-10,150],[-10,200],[-5,0],[-5,50],[-5,100],[-5,150],[-5,200],[0,0],[0,50],[0,100],[0,150],[0,200],[18,0],[18,50],[18,100],[18,150],[18,200],[15,0],[15,50],[15,100],[15,150],[15,200],[10,0],[10,50],[10,100],[10,150],[10,200],[5,0],[5,50],[5,100],[5,150],[5,200]]
            else:
                    self.action_space = range(env.action_space.n)

            self.action_size = len(self.action_space)
            self.observation_shape = self.env.observation_space.shape

            self.state = None
            self.game_name = game_name

    def preprocess(self, obs, is_start=False):
        grayscale = obs.astype('float32').mean(2)
        s = imresize(grayscale, (self.w, self.h)).astype('float32') * (1.0/255.0)
        s = s.reshape(1, s.shape[0], s.shape[1], 1)
        if is_start or self.state is None:
            self.state = np.repeat(s, self.num_frames, axis=3)
        else:
            self.state = np.append(s, self.state[:,:,:,:self.num_frames-1], axis=3)
        return self.state

    def render(self):
        self.env.render()

    def reset(self):
        with verrou1:
            if self.atari == True:
                return self.preprocess(self.env.reset(), is_start=True)
            else:
                return self.env.reset()


    def step(self, action_idx):
        if self.atari:
            action = self.action_space[action_idx]
            accum_reward = 0
            prev_s = None
            for _ in range(self.skip_actions):
                s, r, term, info = self.env.step(action)
                accum_reward += r
                if term:
                    break
                prev_s = s
                # USELESS tho 
            if self.game_name == 'SpaceInvaders-v0' and prev_s is not None:
                s = np.maximum.reduce([s, prev_s])
            elif self.game_name == 'Phoenix-v0' and prev_s is not None:
                s = np.maximum.reduce([s, prev_s])
            elif self.game_name == 'Boxing-v0' and prev_s is not None:
                s = np.maximum.reduce([s, prev_s])
            elif self.game_name == 'Tennis-v0' and prev_s is not None:
                s = np.maximum.reduce([s,prev_s])
            elif self.game_name == 'Skiing-v0' and prev_s is not None:
                s = np.maximum.reduce([s,prev_s])
            return self.preprocess(s), accum_reward, term, info
        else:

            
            action = self.action_space[action_idx]
            accum_reward = 0
            for _ in range(self.skip_actions):
                s, r, term, info = self.env.step(action,0)
                accum_reward += r
                if term:
                    break


            return np.reshape(s, [1, -1]), accum_reward, term, info
