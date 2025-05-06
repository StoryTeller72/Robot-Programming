import gym
import numpy as np

import os
from collections import deque
import random
from project.Reacher.utils.DDPG_Reacher import DDPG
import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import time

current_time = datetime.datetime.now()
writer = SummaryWriter("/home/rustam/PR/project/Reacher/runs")
directory_name = f"/home/rustam/PR/project/Reacher/DDPG_checkpoints/{current_time}"

device = 'cpu'


bath_size = 512
episode_num = 10
trajectory_len = 50
start_learning = 1_000
buffer_size = 200_000

experience_buffer = deque(maxlen=buffer_size)
env = gym.make('Reacher-v4', render_mode='human')
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[0]
action_min = -1.0
action_max = 1.0

agent = DDPG(obs_shape, action_shape, action_min=action_min, action_max=action_max,
             noise_decrease=1 / (episode_num * trajectory_len), polyak=0.05, critic_lr=1e-3, actor_lr=1e-3, noise_scaler=0.8, task_name='DDPG_base')

agent.load_models(
    '/home/rustam/PR/project/Reacher/DDPG_checkpoints/HER/DDPG_HER_not_terminated29')
agent.device = device
agent.policy.to(device)
agent.policy_target.to(device)
agent.Q_fun.to(device)
agent.Q_fun_target.to(device)

step = 0


success_amnt_in_epoch = 0
for episode in range(episode_num):
    observation, info = env.reset()

    state = observation

    for _ in range(trajectory_len):
        action = agent.get_action(state).detach().numpy()
        next_observation, reward, terminated, _, _ = env.step(
            action)

        distance_vector = state[8:10]
        distance = np.linalg.norm(distance_vector)
        next_state = next_observation

        step += 1
        state = next_state
        time.sleep(0.05)
