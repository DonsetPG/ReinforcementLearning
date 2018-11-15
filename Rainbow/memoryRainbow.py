import random
import numpy as np
from collections import deque

############

class Replay():

    def __init__(self,t,state,action,reward,terminal):
        self.t = t
        self.state = state 
        self.action = action 
        self.reward = reward 
        self.terminal = terminal 

############

## Version outdated, needs to be completely reviewed
## it is just slow haha 

## ================================================================ ## 
## ================================================================ ## 

class ReplayMemory():

  def __init__(self, args, capacity):
    self.capacity = capacity
    self.history = args.history_length
    self.discount = args.discount
    self.n = args.multi_step
    self.priority_exponent = args.priority_exponent
    self.t = 0  # Internal episode timestep counter
    self.memory = deque(maxlen=capacity)  
    self.prioritisedScore = deque(maxlen=capacity)
    self.currentMemory = []
    self.currentPrioritisedScore = []
    self.sum = 0.0000001

    # Adds state_(t) and action_(t), reward_(t+1) and terminal_(t+1) 
    def append(self, state, action, reward, terminal):
        t = 0 if terminal else self.t + 1
        replay = Replay(t,state,action,reward,terminal)
        self.t = t
        if not terminal:
            self.currentMemory.append(replay)  # Store new transition with maximum priority
        else:
            self.transferMemory()

    def appendReplayScore(self,err,terminal):
        if not terminal:
            self.currentPrioritisedScore.append(err)
        else:
            self.transferScore(err)

    def transferMemory(self):
        for replay in self.currentMemory:
            self.memory.append(replay)
        self.currentMemory = []

    def transferScore(self,err):
        for i in range(len(self.currentMemory),-1):
            replay = self.currentMemory[i]
        
            target_reward = replay.reward

            new_n = min(self.n,len(self.currentMemory)-1-i)
            for j in range(new_n):
                target_reward += self.discount**(j+1)*self.currentMemory[i+j].reward

            replay.reward = target_reward 
            self.currentPrioritisedScore[i] = (abs(target_reward + self.discount**(new_n+1)*self.currentPrioritisedScore[i]))**(self.priority_exponent)
            
        for score in self.currentPrioritisedScore:
            self.prioritisedScore.append(score)
        self.currentPrioritisedScore = []

    def normalizeScore(self):
        somme = 0
        for i in range(len(self.prioritisedScore)):
            self.prioritisedScore[i] *= self.sum 
            somme += self.prioritisedScore[i]
        for i in range(len(self.prioritisedScore)):
            self.prioritisedScore[i] /= somme
        self.sum = somme

    def sampleBatch(self,batch_size):
        batch = np.random.choice(self.memory,batch_size,p=self.prioritisedScore)
        return batch


    

    

