import os
import sys, getopt
import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import queue
import random

#################

from time import time, sleep, gmtime, strftime
from custom_env import CustomGym
from agent_ppo import Agent
from parametres import get_args_ppo

################

######################################################################################

# Basically a copy of the A3C file, but with the needed modifications to build a PPO algorithms
# the same way openAI implemented it. 


#######################################################################################

random.seed(100)

##############

# Parameters : 

param = get_args_ppo()
T_MAX = param.T_MAX
NUM_THREADS = param.NUM_THREADS
INITIAL_LEARNING_RATE = param.INITIAL_LEARNING_RATE
DISCOUNT_FACTOR = param.DISCOUNT_FACTOR
VERBOSE_EVERY = param.VERBOSE_EVERY
TESTING = param.TESTING
I_ASYNC_UPDATE = param.I_ASYNC_UPDATE
LAMBDA = param.LAMBDA

# Computing flags : 

FLAGS = {"T_MAX": T_MAX, "NUM_THREADS": NUM_THREADS, "INITIAL_LEARNING_RATE":
INITIAL_LEARNING_RATE, "DISCOUNT_FACTOR": DISCOUNT_FACTOR, "VERBOSE_EVERY":
VERBOSE_EVERY, "TESTING": TESTING, "I_ASYNC_UPDATE": I_ASYNC_UPDATE, "lambda" : LAMBDA}

training_finished = False



def async_trainer(agent, env, sess, thread_idx, T_queue, summary, saver,
    save_path):
    print("Training thread", thread_idx)
    T = T_queue.get()
    T_queue.put(T+1)
    t = 0

    last_verbose = T
    last_time = time()
    last_target_update = T

    terminal = True
    while T < T_MAX:
        t_start = t
        batch_states = []
        batch_rewards = []
        batch_actions = []
        baseline_values = []
        batch_delta = []

        if terminal:
            terminal = False
            #print("Training thread ended", thread_idx)
            state = env.reset()

        while not terminal and len(batch_states) < I_ASYNC_UPDATE:
   
            batch_states.append(state)
            policy, value = agent.get_policy_and_value(state)
            action_idx = np.random.choice(agent.action_size, p=policy)
            state, reward, terminal, _ = env.step(action_idx)
            next_value = agent.get_value(state)  
            t += 1
            T = T_queue.get()
            T_queue.put(T+1)
            reward = np.clip(reward, -1, 1)

            #Calcul delta (from the PPO paper)

            delta = reward + DISCOUNT_FACTOR*next_value[0] - value[0]

 
            batch_delta.append(delta)
            batch_rewards.append(reward)
            batch_actions.append(action_idx)
            baseline_values.append(value[0])


        target_value = 0
      
        if not terminal:
            target_value = agent.get_value(state)[0]
        last_R = target_value

      
        batch_target_values = []
        for reward in reversed(batch_rewards):
            target_value = reward + DISCOUNT_FACTOR * target_value
            batch_target_values.append(target_value)
        batch_target_values.reverse()

        ## Compute the sampled n-step discounted delta
        batch_advantages = []
        delta_bis = 0

        for delta in reversed(batch_delta):
            delta_bis = delta + DISCOUNT_FACTOR * LAMBDA * delta_bis
            batch_advantages.append(delta_bis)

        batch_advantages.reverse()
        agent.train(np.vstack(batch_states), batch_actions, batch_target_values,
        batch_advantages)

    global training_finished
    training_finished = True

def estimate_reward(agent, env, episodes=10, max_steps=10000):
    episode_rewards = []
    episode_vals = []
    t = 0
    for i in range(episodes):
        episode_reward = 0
        state = env.reset()
        terminal = False
        while not terminal:
            policy, value = agent.get_policy_and_value(state)
            action_idx = np.random.choice(agent.action_size, p=policy)

            state, reward, terminal, _ = env.step(action_idx)
            t += 1
            episode_vals.append(value)
            episode_reward += reward
            if t > max_steps:
                episode_rewards.append(episode_reward)
                return episode_rewards, episode_vals
        episode_rewards.append(episode_reward)
    return episode_rewards, episode_vals

def evaluator(agent, env, sess, T_queue, summary, saver, save_path):
   
    T = T_queue.get()
    T_queue.put(T)
    last_time = time()
    last_verbose = T
    while T < T_MAX:
        T = T_queue.get()
        T_queue.put(T)
        if T - last_verbose >= VERBOSE_EVERY:
            print("Current T => ", T)
            current_time = time()
            print("Train steps per second (T/s) ==> ", float(T - last_verbose) / (current_time - last_time))
            last_time = current_time
            last_verbose = T

            print("Evaluating agent...")
            episode_rewards, episode_vals = estimate_reward(agent, env, episodes=5)
            avg_ep_r = np.mean(episode_rewards)
            avg_val = np.mean(episode_vals)
            print("Average reward", avg_ep_r, "Average value", avg_val)

            summary.write_summary({'episode_avg_reward': avg_ep_r, 'avg_value': avg_val}, T)
            checkpoint_file = saver.save(sess, save_path, global_step=T)
            print("Saved in", checkpoint_file)
        sleep(1.0)


def ppo(game_name, num_threads=NUM_THREADS, restore=None, save_path='model'):
    processes = []
    envs = []
    atari = 'not mnih'
    for _ in range(num_threads+1):
        
        if game_name == 'Csb-only-runner-v0':
            print("Playing with ",game_name," ")
            env = CustomGym(game_name)
        else:
            print("Playing with ATARI")
            atari = 'mnih'
            gym_env = gym.make(game_name)
            env = CustomGym(game_name)
        envs.append(env)

   
    evaluation_env = envs[0]
    envs = envs[1:]

    with tf.Session() as sess:
        agent = Agent(session=sess,
        action_size=envs[0].action_size, model=atari,
        optimizer=tf.train.AdamOptimizer(INITIAL_LEARNING_RATE))

        saver = tf.train.Saver(max_to_keep=2)

        T_queue = queue.Queue()
        S_queue = []
        
        if restore is not None:
            saver.restore(sess, save_path + '-' + str(restore))
            last_T = restore
            print("T was:", last_T)
            T_queue.put(last_T)
        else:
            sess.run(tf.global_variables_initializer())
            T_queue.put(0)

        summary = Summary(save_path, agent)

       
        for i in range(num_threads):
            processes.append(threading.Thread(target=async_trainer, args=(agent,
            envs[i], sess, i, T_queue, summary, saver, save_path)))

       
        processes.append(threading.Thread(target=evaluator, args=(agent,
        evaluation_env, sess, T_queue, summary, saver, save_path,)))

       
        for p in processes:
            p.daemon = True
            p.start()

       
        while not training_finished:
            sleep(0.01)

       
        for p in processes:
            p.join()


def discount(rewards, gamma):
    return np.sum([rewards[i] * gamma**i for i in range(len(rewards))])

def test_equals(arr1, arr2, eps):
    return np.sum(np.abs(np.array(arr1)-np.array(arr2))) < eps

class Summary:
    def __init__(self, logdir, agent):
        with tf.variable_scope('summary'):
            summarising = ['episode_avg_reward', 'avg_value']
            self.agent = agent
            self.writer = tf.summary.FileWriter(logdir, self.agent.sess.graph)
            self.summary_ops = {}
            self.summary_vars = {}
            self.summary_ph = {}
            for s in summarising:
                self.summary_vars[s] = tf.Variable(0.0)
                self.summary_ph[s] = tf.placeholder('float32', name=s)
                self.summary_ops[s] = tf.summary.scalar(s, self.summary_vars[s])
            self.update_ops = []
            for k in self.summary_vars:
                self.update_ops.append(self.summary_vars[k].assign(self.summary_ph[k]))
            self.summary_op = tf.summary.merge(list(self.summary_ops.values()))

    def write_summary(self, summary, t):
        self.agent.sess.run(self.update_ops, {self.summary_ph[k]: v for k, v in summary.items()})
        summary_to_add = self.agent.sess.run(self.summary_op, {self.summary_vars[k]: v for k, v in summary.items()})
        self.writer.add_summary(summary_to_add, global_step=t)

def main(argv):
    num_threads = NUM_THREADS
    game_name = 'SpaceInvaders-v0'
    save_path = None
    restore = None
    try:
        opts, args = getopt.getopt(argv, "hg:s:r:t:")
    
    for opt, arg in opts:
        if opt == '-g':
            game_name = arg
        elif opt == '-s':
            save_path = arg
        elif opt == '-r':
            restore = int(arg)
        elif opt == '-t':
            num_threads = int(arg)
            print("Using", num_threads, "threads.")
    if game_name is None:
        print("No game name specified")
        sys.exit()
    if save_path is None:
        save_path = 'experiments/' + game_name + '/' + \
        strftime("%d-%m-%Y-%H:%M:%S/model", gmtime())
        print("so saving to", save_path)
    if not os.path.exists(save_path):
        print("Path doesn't exist, so creating")
        os.makedirs(save_path)
    print("Using save path", save_path)
    print("Using flags", FLAGS)
    ppo(game_name, num_threads=NUM_THREADS, restore=restore,
    save_path=save_path)

if __name__ == "__main__":
    main(sys.argv[1:])
