from custom_env import CustomGym
from agent import Agent
import os, getopt, sys
import gym
import tensorflow as tf
import numpy as np
from time import time, sleep


#/Users/paulgarnier/Desktop/Files/GitHub/RL/experiments/Phoenix-v0/08-10-2018-14:30:12/model
#T = 9967476
#python run_agent.py -g "Phoenix-v0" -s "/Users/paulgarnier/Desktop/Files/GitHub/RL/experiments/Phoenix-v0/08-10-2018-14:30:12/model" -T "9967476"

#/Users/paulgarnier/Desktop/Files/GitHub/RL/experiments/Tennis-v0/09-10-2018-17:11:21/model
#T = 5208256
#python run_agent.py -g "Tennis-v0" -s "/Users/paulgarnier/Desktop/Files/GitHub/RL/experiments/Tennis-v0/09-10-2018-17:11:21/model" -T "5208256"

#/Users/paulgarnier/Desktop/Files/GitHub/RL/experiments/Boxing-v0/10-10-2018-10:50:17/model
#T = 2219073
#python run_agent.py -g "Boxing-v0" -s "/Users/paulgarnier/Desktop/Files/GitHub/RL/experiments/Boxing-v0/10-10-2018-10:50:17/model" -T "2219073"

#/Users/paulgarnier/Desktop/Files/GitHub/RL/experiments/Skiing-v0/12-10-2018-22:08:27/model
#T = 8181790
#python run_agent.py -g "Skiing-v0" -s "/Users/paulgarnier/Desktop/Files/GitHub/RL/experiments/Skiing-v0/12-10-2018-22:08:27/model" -T "8181790"




def run_agent(save_path, T, game_name):
    with tf.Session() as sess:
        agent = Agent(session=sess,action_size=18)
        saver = tf.train.Saver()

        saver.restore(sess, save_path + '-' + str(T))

        play(agent, game_name)

        return sess, agent

def play(agent, game_name, render=True, num_episodes=5, fps=5.0, monitor=True):
    env = CustomGym(game_name)

    desired_frame_length = 1.0 /15 

    episode_rewards = []
    episode_vals = []
    t = 0
    for ep in range(num_episodes):
        print("Starting episode", ep)
        episode_reward = 0
        state = env.reset()
        terminal = False
        current_time = time()
        while not terminal:
            policy, value = agent.get_policy_and_value(state)
            action_idx = np.random.choice(agent.action_size, p=policy)
            state, reward, terminal, _ = env.step(action_idx)
            if render:
                env.render()
                sleep(0.1)
            t += 1
            episode_vals.append(value)
            episode_reward += reward
            next_time = time()
            frame_length = next_time - current_time
            if frame_length < desired_frame_length:
                sleep(desired_frame_length - frame_length)
            current_time = next_time
        episode_rewards.append(episode_reward)
  
    return episode_rewards, episode_vals

def main(argv):
    save_path = None
    T = None
    game_name = None
    try:
        opts, args = getopt.getopt(argv, "g:s:T:")
    for opt, arg in opts:
        if opt == '-g':
            game_name = arg
        elif opt == '-s':
            save_path = arg
        elif opt == '-T':
            T = arg
    if game_name is None:
        print("No game name")
        sys.exit()
    if save_path is None:
        print("No save path")
        sys.exit()
    if T is None:
        print("No T")
        sys.exit()
    print("Reading from", save_path)
    print("Running agent")
    run_agent(save_path, T, game_name)

if __name__ == "__main__":
    main(sys.argv[1:])
