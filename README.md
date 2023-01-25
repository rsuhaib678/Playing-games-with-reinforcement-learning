# Playing-games-with-reinforcement-learning

Video Link: https://youtu.be/bZi8P6RNagU

Created agent to play games using different reinforcement learning techniques

Before executing the code, the following Python libraries need to be installed with a Python version of 3.7:
- mlagents
- pytorch
- tensorflow
- numpy
- wandb
- ipdb

For issues with tensorflow, activate a virtual environment through conda and install the following libraries:
- mlagents
- wandb
- ipdb

To train :
- single agent using DQN algorithm, change the working directory to  /submission/DQN and execute the Python script dqn_exp_unity_v3.py.

- 16 agents using DQN algorithm, change the working directory to /submission/MultiagentDQN and execute the Python script dqn_exp_unity_multiagent_v2.py.

- Agent with PPO algorithm, change the working directory to /submission/PPO and execute the Python script ppo_unity.py.


After executing the script, when prompted, follow the steps displayed on the console to create a profile on wandb. We are using wandb to produce the results in a graph format continuously.
The results can be seen through the wandb URL displayed after executing the command.
