# RL

Several RL algorithms : 

- A3C : https://arxiv.org/abs/1602.01783
- PPO : https://blog.openai.com/openai-baselines-ppo/ && https://arxiv.org/abs/1707.06347
- QL : https://arxiv.org/abs/1312.5602

Implementations are using Keras or Tensorflow, and the environments are Gym based (https://gym.openai.com/envs/#classic_control) or handmade, inspired from Codingame (https://www.codingame.com/home). 

 - My first attempt was based on Coder Strikes Back (https://www.codingame.com/multiplayer/bot-programming/coders-strike-back). The algorithm used is deep Q-learning, with several improvements : an implementation of Hindsight Experience Replay (HER), based on (https://arxiv.org/abs/1707.01495), and a gestion of the memory based on how good the reward is. 
 
 - After that, I used openAI implementation of A3C, and several projects on GitHub to make my own implementation of A3C, working with gym and handmade environments. An example of an Agent on codingame : https://www.codingame.com/replay/349814097. (My running bot on the website is not currently using RL but a Genetic Algorithm I am currently working on, since it is based on another players code for the most part)
 
 - I am currently working on implementing PPO, and a new kind of RL algorithm, using MCTS to help the action space exploration (in addition to the entropy we use in the policy exploration). 
