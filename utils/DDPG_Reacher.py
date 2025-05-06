import copy
import numpy as np
import torch
import torch.nn as nn
import random
import os
import datetime


class Actor(nn.Module):
    def __init__(self, observation_space_dim,  action_space_dim, layer_1_dim=400, layer_2_dim=300):
        super().__init__()
        self.layer_1 = nn.Linear(observation_space_dim, layer_1_dim)
        self.layer_2 = nn.Linear(layer_1_dim, layer_2_dim)
        self.layer_3 = nn.Linear(layer_2_dim, action_space_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        hidden = self.layer_1(input)
        hidden = self.relu(hidden)
        hidden = self.layer_2(hidden)
        hidden = self.relu(hidden)
        output = self.layer_3(hidden)
        return self.tanh(output)


class Critic(nn.Module):
    def __init__(self, observation_space_dim,  action_space_dim, layer_1_dim=400, layer_2_dim=300):
        super().__init__()
        self.layer_1 = nn.Linear(
            observation_space_dim + action_space_dim, layer_1_dim)
        self.layer_2 = nn.Linear(layer_1_dim, layer_2_dim)
        self.layer_3 = nn.Linear(layer_2_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, input):
        hidden = self.layer_1(input)
        hidden = self.relu(hidden)
        hidden = self.layer_2(hidden)
        hidden = self.relu(hidden)
        return self.layer_3(hidden)


class DDPG():
    def __init__(self, observation_space_dim,  action_space_dim, noise_decrease, action_min, action_max, grad_steps=1, polyak=1e-2, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, noise_scaler=1.0, sigma=0.3, train_mode=True, task_name=None):
        self.observation_dim = observation_space_dim
        self.action_dim = action_space_dim
        self.action_min = action_min
        self.action_max = action_max
        self.grad_steps = grad_steps
        self.device = 'cpu'
        self.train_mode = train_mode
        self.task_name = task_name
        self.policy = Actor(self.observation_dim, self.action_dim)
        self.policy_target = copy.deepcopy(self.policy)
        self.Q_fun = Critic(self.observation_dim, self.action_dim)
        self.Q_fun_target = copy.deepcopy(self.Q_fun)
        self.actor_lr = actor_lr

        # torch.nn.init.normal_(self.policy, mean=0, std=0.1)
        # torch.nn.init.normal_(self.policy_target, mean=0, std=0.1)
        # torch.nn.init.normal_(self.Q_fun, mean=0, std=0.1)
        # torch.nn.init.normal_(self.Q_fun_target, mean=0, std=0.1)

        self.critic_lr = critic_lr

        self.polyak = polyak
        self.noise_scaler = noise_scaler
        self.noise_decrease = noise_decrease

        self.sigma = sigma

        self.gamma = gamma
        self.q_optimizer = torch.optim.Adam(
            self.Q_fun.parameters(), lr=critic_lr)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=actor_lr)

    def get_action(self, state):
        state_t = torch.FloatTensor(state).to(self.device)
        action = self.policy(state_t).to(self.device)
        if self.train_mode:
            action += torch.normal(0, self.sigma,
                                   (1, self.action_dim)).squeeze().to(self.device)
        action = torch.clamp(action, self.action_min, self.action_max)
        self.noise_scaler = max(0, self.noise_decrease - self.noise_decrease)
        action = torch.clamp(action, self.action_min, self.action_max)
        action = action.to('cpu')
        return action

    def update(self, target_model, model, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(
                (1 - self.polyak) * target_param.data + self.polyak * param.data)

    def fit(self, bath):
        states, actions, rewards, done, next_states = map(
            torch.FloatTensor, zip(*bath))
        rewards = rewards.reshape(len(bath), 1)
        done = done.reshape(len(bath), 1)
        states_t = states.to(self.device)
        actions_t = actions.to(self.device)
        rewards_t = rewards.to(self.device)
        done_t = done.to(self.device)
        next_states_t = next_states.to(self.device)

        # Calculate Loss for critic (Q_function) and update
        pred_next_actions = self.policy_target(next_states_t)
        pred_next_actions_t = pred_next_actions.to(self.device)
        next_states_and_actions = torch.cat(
            (next_states_t, pred_next_actions_t), dim=1)

        next_states_and_actions_t = next_states_and_actions.to(self.device)
        targets = rewards_t + self.gamma * \
            (1 - done_t) * self.Q_fun_target(next_states_and_actions_t)
        states_and_action = torch.cat((states_t, actions_t), dim=1)

        # states_and_action.to(self.device)
        Q_loss = torch.mean(
            (targets.detach() - self.Q_fun(states_and_action))**2)
        self.update(self.Q_fun_target, self.Q_fun,
                    self.q_optimizer, Q_loss)

        # Calculate Loss for actor(policy) and update
        pred_action = self.policy(states_t)
        states_and_pred_actions = torch.cat((states_t, pred_action), dim=1)
        # states_and_pred_actions.to(self.device)
        policy_loss = -torch.mean(self.Q_fun(states_and_pred_actions))

        self.update(self.policy_target, self.policy,
                    self.policy_optimizer, policy_loss)

        states.to('cpu')
        actions.to('cpu')
        rewards.to('cpu')
        done.to('cpu')
        next_states.to('cpu')
        next_states_and_actions.to('cpu')
        states_and_action.to('cpu')
        states_and_pred_actions.to('cpu')
        return (policy_loss, Q_loss)

    def save_models(self, path, check_point_number):
        torch.save({
            'policy_model': self.policy.state_dict(),
            'target_policy_model': self.policy_target.state_dict(),
            'Q_fun_model': self.Q_fun.state_dict(),
            'Q_fun_target_model': self.Q_fun_target.state_dict()

        }, path + '/'+self.task_name + str(check_point_number))

    def load_models(self, PATH):
        checkpoint = torch.load(PATH, weights_only=True)
        self.policy.load_state_dict(checkpoint['policy_model'])
        self.policy_target.load_state_dict(checkpoint['target_policy_model'])
        self.Q_fun.load_state_dict(checkpoint['Q_fun_model'])
        self.Q_fun.load_state_dict(checkpoint['Q_fun_target_model'])
