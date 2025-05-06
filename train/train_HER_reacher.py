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
writer = SummaryWriter("/home/rustam/PR/project/Reacher/runs")
directory_name = f"/home/rustam/PR/project/Reacher/DDPG_checkpoints/HER_60"
os.makedirs(directory_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


bath_size = 512
episode_num = 100
num_epoch = 60
trajectory_len = 50
start_learning = 1_000
buffer_size = 1_000_000
step = 0

experience_buffer = deque(maxlen=buffer_size)
env = gym.make('Reacher-v4')
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[0]
action_min = -1.0
action_max = 1.0
success_rate_array = np.zeros(num_epoch)

agent = DDPG(obs_shape, action_shape, action_min=action_min, action_max=action_max,
             noise_decrease=1 / (num_epoch * episode_num * trajectory_len), polyak=0.005, critic_lr=1e-3, actor_lr=1e-3, noise_scaler=0.3, task_name='DDPG_HER_not_terminated_60')

agent.device = device
agent.policy.to(device)
agent.policy_target.to(device)
agent.Q_fun.to(device)
agent.Q_fun_target.to(device)


for epoch in range(num_epoch):
    success_amnt_in_epoch = 0
    for episode in range(episode_num):
        observation, info = env.reset()
        goals = []
        state = observation
        cur_episode_trajectory = {}
        for t in range(trajectory_len):
            action = agent.get_action(state).detach().numpy()
            next_observation, _, _, _, _ = env.step(action)
            distance_vector = state[8:10]
            distance = np.linalg.norm(distance_vector)
            reward = 1 if distance < 0.02 else 0
            next_state = next_observation
            cur_episode_trajectory[t] = (
                state, action, reward, False, next_state)
            goals.append(state[4:6])
            state = next_state
            if t == trajectory_len - 1 and reward:
                success_amnt_in_epoch += 1
            step += 1
        for t in range(len(cur_episode_trajectory)):
            experience_buffer.append(cur_episode_trajectory[t])
            extra_gols = random.sample(
                goals[t:], 5) if len(goals[t:]) > 5 else []
            cur_state, cur_action, _, _, cur_next_state = cur_episode_trajectory[t]
            for g in extra_gols:
                cur_distance = cur_state[4:6] + cur_state[8:10] - g
                cur_distance_next = cur_next_state[4:6] + \
                    cur_next_state[8:10] - g
                cur_goal = g
                cur_reward = 1 if np.linalg.norm(
                    cur_distance_next) < 0.02 else 0
                cur_next_state[4] = g[0]
                cur_next_state[5] = g[1]
                cur_next_state[8] = cur_distance_next[0]
                cur_next_state[9] = cur_distance_next[1]

                cur_state[4] = g[0]
                cur_state[5] = g[1]
                cur_state[8] = cur_distance[0]
                cur_state[9] = cur_distance[1]

                experience_buffer.append((
                    cur_state, cur_action, cur_reward, False, cur_next_state))

        if step > start_learning:
            for _ in range(10):
                bath = random.sample(experience_buffer, bath_size)
                policy_loss, Q_los = agent.fit(bath)

    success_rate = success_amnt_in_epoch / episode_num
    success_rate_array[epoch] = success_rate

    print('epoch:', epoch, 'success_rate:', success_rate)
    writer.add_scalar('success_rate', success_rate, epoch)
    writer.add_scalar('Q_loss', Q_los, epoch)
    writer.add_scalar('Pi_Loss', policy_loss, epoch)
    if (epoch + 1) % 10 == 0:
        agent.save_models(directory_name, epoch)
np.save('DDPG_HER_not_terminated_60', success_rate_array)
writer.close()
