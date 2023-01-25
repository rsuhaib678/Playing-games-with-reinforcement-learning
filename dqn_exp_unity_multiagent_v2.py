from typing import Any
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from random import random, sample
import wandb
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.environment import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)

@dataclass
class Sarsd:
    state: Any
    action : int
    reward : float
    next_state : Any
    done: bool

class Agent :
    def __init__(self, model):
        self.model = model

    #apply the model to this obsvation and get Q value
    def act(self, obsvations):
        #_observation share is (N, 4)
        q_vals = self.model(obsvations)

        #q_vals shape (N, 2)
        return q_vals.max(-1)

class HighLevelModel(nn.Module): 
    def __init__(self, obs_shape):
        super(HighLevelModel,self).__init__()
        # import ipdb; ipdb.set_trace();
        # assert len(obs_shape) ==1, "This network only works for flat observations"
        self.net = nn.Sequential(
            nn.Linear(obs_shape, 256),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class LowLevelNN(nn.Module):
    def __init__(self, num_outputs) -> None:
        super(LowLevelNN, self).__init__()
        self.fc1 = nn.Linear(256, num_outputs)

    def forward(self,x):
        # make input flow the network
        return self.fc1(x)

class Model(nn.Module):
    def __init__(self, obs_shape, num_actions ) -> None:
        super(Model,self).__init__()
        self.num_actions = num_actions
        self.obs_shape = obs_shape
        # High level nn perform all initial operation
        self.net = HighLevelModel(obs_shape)
        # Low level action specific nn
        self.outputs=[None] * len(num_actions)
        for action_index in range(len(num_actions)):
            self.outputs[action_index] = LowLevelNN(num_actions[action_index])
        self.opt = optim.Adam(self.parameters(),lr=0.0001)
    
    def forward(self,x):
        x = self.net(x)
        # for action_index in range(len(self.num_actions)):
        #     x = self.outputs[action_index](x)
        return [self.outputs[action_index](x) for action_index in range(len(self.num_actions))]

class ReplayBuffer:
    def __init__(self, buffer_size=100000):
        self.buffer_size = buffer_size
        self.buffer = [None] * buffer_size
        self.idx = 0
        # self.buffer = deque(maxlen=buffer_size)

    def insert(self, sars):
        self.buffer[self.idx%self.buffer_size] =  sars
        self.idx += 1

    def sample(self, num_samples):
        # assert num_samples <= len(self.buffer)
        assert num_samples < max(self.idx, self.buffer_size)
        if self.idx < self.buffer_size :
            return sample(self.buffer[:self.idx], num_samples)
        return sample(self.buffer, num_samples)

def update_tgt_model(m, tgt):
    tgt.load_state_dict(m.state_dict())

def train_step(model, state_transitions, tgt, gamma=0.99):
    # Create state vector
    cur_states = torch.stack([torch.Tensor(s.state) for s in state_transitions])
    rewards = torch.stack([torch.Tensor([s.reward]) for s in state_transitions])
    mask = torch.stack([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])
    next_states = torch.stack([torch.Tensor(s.next_state) for s in state_transitions])
    actions = np.array([s.action for s in state_transitions])

    next_obs = tgt(next_states)
    qvals = model(cur_states)
    qvals_next_array=[0]*len(model.num_actions)
    
    with torch.no_grad():
        for index in range(len(model.num_actions)):
            qvals_next_array[index] = next_obs[index].max(-1)[0]
    
    # import ipdb; ipdb.set_trace();
    qvals_next = torch.cat(qvals_next_array, dim=1)
    model.opt.zero_grad()
    # qvals = cur_pred[index]
    one_hot_actions = [0]*len(model.num_actions)
    pred_qvals = [0]*len(model.num_actions)
    # import ipdb; ipdb.set_trace();
    for index in range(len(model.num_actions)):
        one_hot_actions[index] = F.one_hot(torch.LongTensor(actions[:,0,index]), model.num_actions[index])
        pred_qvals[index] = torch.sum(qvals[index].reshape(qvals[index].shape[0],model.num_actions[index]) * one_hot_actions[index], -1)
    # loss = ((rewards + mask[:,0]*qvals_next.sum(dim=1) * gamma - np.sum(pred_qvals).reshape(750,1))**2).mean()
    loss = ((rewards + mask[:,0]*qvals_next.sum(dim=1) * gamma - torch.stack(pred_qvals, dim=0).sum(dim=0).reshape(750,1))**2).mean()
    # if(loss < 10):
    #     import ipdb; ipdb.set_trace();

    loss.backward()
    model.opt.step()
    return loss

def main(test=False, chkpt=None):
    if not test:
        wandb.init(project="dqn-autopilot", name="dqn-autopilot")
    min_rb_size = 5000
    sample_size = 750
    episode_count = 0
    eps_min = 0.1

    eps_decay = 0.99999
    env_steps_before_train = 100
    tgt_model_update = 200

    ENV_NAME = "build"

    engine_config_channel = EngineConfigurationChannel()
    if not test :
        time_scale = 40
        width = 100
        height = 100
    else:
        time_scale = 1
        width = 1800
        height = 900

    engine_config_channel.set_configuration_parameters(
        width=width, height=height, time_scale=time_scale
    )
    env = UnityEnvironment(
        file_name=ENV_NAME, seed=1, side_channels=[engine_config_channel]
    )
    env.reset()
    behavior_name = list(env.behavior_specs)[0]
    print(f"Name of the behavior : {behavior_name}")
    spec = env.behavior_specs[behavior_name]
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    NO_OF_AGENT = len(decision_steps.agent_id)
    last_observation = np.concatenate(decision_steps.obs, axis=1)
    # import ipdb; ipdb.set_trace()
    m = Model(sum([observation.shape[0] for observation in spec.observation_specs]), spec.action_spec.discrete_branches)
    if chkpt is not None:
        m.load_state_dict(torch.load(chkpt))
    tgt = Model(sum([observation.shape[0] for observation in spec.observation_specs]), spec.action_spec.discrete_branches)
    update_tgt_model(m,tgt)

    rb = ReplayBuffer()
    steps_since_train = 0
    steps_since_tgt = 0


    step_num = -1 * min_rb_size
    # qvals = m(torch.Tensor(observation))
    # import ipdb; ipdb.set_trace()

    episode_rewards = []
    rolling_reward = 0

    curriculam_reward_step = [5,5,7,9]
    curriculam_distance_vector = [50,30,20,0]

    tq = tqdm()
    try: 
        while True:
            # if test:
                # env.render()
                # time.sleep(0.05)
            tq.update(1)

            eps = eps_decay**(step_num)
            if eps <= eps_min:
                eps = eps_min                

            if test:
                eps = 0

            action_array = [0]*NO_OF_AGENT
            for agent_id in range(NO_OF_AGENT):
                if random() < eps:
                    # import ipdb; ipdb.set_trace()
                    action = spec.action_spec.random_action(1).discrete
                else:
                    # import ipdb; ipdb.set_trace()
                    action_tensor = m(torch.Tensor(last_observation[agent_id]))
                    action = [0]*len(m.num_actions)
                    for i in range(len(m.num_actions)):
                        action[i] = action_tensor[i].max(-1)[-1].item() 
                    action = np.array([action])
                action_array[agent_id]=action

            # Take action
            actiontuple = ActionTuple()
            actiontuple.add_discrete(np.array(action_array).reshape(16,3))
            env.set_actions(behavior_name, actiontuple)
            env.step()
            # Observe the env
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            # End of Episode
            if(len(terminal_steps.agent_id) > 0):
                # import ipdb; ipdb.set_trace()
                reward = terminal_steps.reward[0]
                rolling_reward += reward
                episode_count += 1
                # reward = reward/100 #TODO
                episode_rewards.append(rolling_reward)
                if test: 
                    print(rolling_reward)
                rolling_reward = 0
                observation = np.concatenate(terminal_steps.obs, axis=1)
                # import ipdb; ipdb.set_trace()
                rb.insert(Sarsd(last_observation[terminal_steps.agent_id[0]].reshape(1,93), action_array[terminal_steps.agent_id[0]], reward, observation[0].reshape(1,93), True))
                env.reset()
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                observation = np.concatenate(decision_steps.obs, axis=1)
            else :
                for agent_id in range(NO_OF_AGENT):
                    reward = decision_steps.reward[agent_id]
                    rolling_reward += reward
                    # reward = reward/100 #TODO
                    observation = np.concatenate(decision_steps.obs, axis=1)
                    # import ipdb; ipdb.set_trace()
                    rb.insert(Sarsd(last_observation[agent_id].reshape(1,93), action_array[agent_id], reward, observation[agent_id].reshape(1,93), False))

            last_observation = observation
            
            steps_since_train += 1
            step_num += 1
            if (not test) and rb.idx > min_rb_size and steps_since_train > env_steps_before_train:
                loss = train_step(m, rb.sample(sample_size), tgt)
                wandb.log({'loss': loss.detach().item(), 'eps':eps , 'avg_reward':np.mean(episode_rewards)}, step=episode_count)
                # Check if mean reward is greater 5 then increse the difficulty
                # wandb.log({'Episode Avg reward':np.mean(episode_rewards)}, step=episode_count)
                # print(step_num, loss.detach().item())
                steps_since_tgt +=1
                if(steps_since_tgt > tgt_model_update):
                    print("Updating target model")
                    update_tgt_model(m, tgt)
                    steps_since_tgt=0
                    torch.save(tgt.state_dict(),f"models/{step_num}.pth")
                steps_since_train = 0
    except KeyboardInterrupt:
        pass
    env.close()

if __name__ == '__main__' :
    # main(True, "models/1050892.pth")
    main()
