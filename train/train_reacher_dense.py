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

current_time = datetime.datetime.now()
writer = SummaryWriter("runs")
directory_name = f"chekpoints/dense_reward/dense_reward_{current_time}"
os.makedirs(directory_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


bath_size = 512
episode_num = 100
num_epoch = 50
trajectory_len = 50
start_learning = 1_000
buffer_size = 1_000_000

experience_buffer = deque(maxlen=buffer_size)
env = gym.make('Reacher-v4')
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[0]
action_min = -1.0
action_max = 1.0

agent = DDPG(obs_shape, action_shape, action_min=action_min, action_max=action_max,
             noise_decrease=1 / (num_epoch * episode_num * trajectory_len), polyak=0.05, critic_lr=2e-3, actor_lr=1e-3, noise_scaler=0.3, task_name='DDPG_base_dense')

agent.device = device
agent.policy.to(device)
agent.policy_target.to(device)
agent.Q_fun.to(device)
agent.Q_fun_target.to(device)

step = 0

mean_reward_array = np.zeros(num_epoch)

for epoch in range(num_epoch):
    epoch_reward = 0
    for episode in range(episode_num):
        cur_episode_reward = 0
        observation, info = env.reset()
        state = observation

        for _ in range(trajectory_len):
            action = agent.get_action(state).detach().numpy()
            next_observation, reward, terminated, _, _ = env.step(
                action)

            next_state = next_observation

            experience_buffer.append(
                (state, action, reward, terminated, next_state))

            step += 1
            state = next_state
            cur_episode_reward += reward

            if len(experience_buffer) > bath_size:
                bath = random.sample(experience_buffer, bath_size)
                policy_loss, Q_los = agent.fit(bath)
                writer.add_scalar('Q_loss', Q_los, episode)
                writer.add_scalar('Pi_Loss', policy_loss, episode)
        epoch_reward += cur_episode_reward
    mean_reward = epoch_reward / episode_num
    mean_reward_array[epoch] = mean_reward
    print('epoch:', epoch, 'mean_reward:', mean_reward)
    writer.add_scalar('mean_reward', mean_reward, epoch)
    writer.add_scalar('Q_loss', Q_los, epoch)
    writer.add_scalar('Pi_Loss', policy_loss, epoch)
    if (epoch + 1) % 10 == 0:
        agent.save_models(directory_name, epoch)
np.save('base_DDPG_dense', mean_reward_array)
writer.close()
