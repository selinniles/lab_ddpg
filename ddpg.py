import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import copy
import numpy as np
from collections import deque
import random
from torch.autograd import Variable
from task_env import secretary_task_env

# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self) -> object:
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


# Ornstein-Ulhenbeck Process
# Taken from https://gist.github.com/cyoon1729/2ea43c5e1b717cc072ebc28006f4c887#file-utils-py
class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class CriticNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_layer_sizes=None):
        super(CriticNN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ln = []  # LayerNormalization

        if hidden_layer_sizes is None:
            self.layers = nn.ModuleList([nn.Linear(input_size, output_size)])

        else:
            self.layers = nn.ModuleList([nn.Linear(input_size, hidden_layer_sizes[0])])
            self.ln.append(nn.LayerNorm(hidden_layer_sizes[0]).to(self.device))
            for i in range(len(hidden_layer_sizes) - 1):
                self.layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))
                self.ln.append(nn.LayerNorm(hidden_layer_sizes[i+1]).to(self.device))
            self.layers.append(nn.Linear(hidden_layer_sizes[len(hidden_layer_sizes) - 1], output_size))

    def forward(self, state, action):

        if len(self.layers) == 1:
            x = torch.cat([state, action], 1)
            return self.layers[0](x)

        else:
            x = torch.cat([state, action], 1)
            for i in range(len(self.layers) - 1):
                x = F.relu(self.layers[i](x))
                # x = self.ln[i](x)
            return self.layers[len(self.layers) - 1](x)


class ActorNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_layer_sizes=None):
        super(ActorNN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ln = []  # LayerNormalization
        self.bn = []  # BatchNormalization

        if hidden_layer_sizes is None:
            self.layers = nn.ModuleList([nn.Linear(input_size, output_size)])

        else:
            self.layers = nn.ModuleList([nn.Linear(input_size, hidden_layer_sizes[0])])
            self.ln.append(nn.LayerNorm(hidden_layer_sizes[0]).to(self.device))
            self.bn.append(nn.BatchNorm1d(hidden_layer_sizes[0]).to(self.device))
            for i in range(len(hidden_layer_sizes) - 1):
                self.layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))
                self.ln.append(nn.LayerNorm(hidden_layer_sizes[i+1]).to(self.device))
                self.bn.append(nn.BatchNorm1d(hidden_layer_sizes[i+1]).to(self.device))
            self.layers.append(nn.Linear(hidden_layer_sizes[len(hidden_layer_sizes) - 1], output_size))

    def forward(self, state):

        if len(self.layers) == 1:
            return torch.tanh(self.layers[0](state))

        elif len(state.shape) == 1:
            for i in range(len(self.layers) - 1):
                state = F.relu(self.layers[i](state))
                # state = self.ln[i](state)
            return torch.tanh(self.layers[len(self.layers) - 1](state))
        else:
            for i in range(len(self.layers) - 1):
                state = F.relu(self.layers[i](state))
                # state = self.bn[i](state)
            return torch.tanh(self.layers[len(self.layers) - 1](state))


# Thanks Chris Yoon -> https://gist.github.com/cyoon1729/542edc824bfa3396e9767e3b59183cae#file-ddpg-py
class Ddpg:

    def __init__(self, state_dim, action_dim, actor_hid, critic_hid, actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, tau=1e-2, replay_size=16384, batch_size=128):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.memory = Memory(replay_size)
        self.discount = gamma
        self.tau = tau

        # The 4 Networks
        self.Actor = ActorNN(input_size=state_dim, output_size=action_dim, hidden_layer_sizes=actor_hid).to(self.device)
        self.Target_Actor = copy.deepcopy(self.Actor).to(self.device)
        self.Critic = CriticNN(input_size=state_dim + action_dim, output_size=action_dim, hidden_layer_sizes=critic_hid).to(self.device)
        self.Target_Critic = copy.deepcopy(self.Critic).to(self.device)

        self.Actor_optimizer = optim.Adam(self.Actor.parameters(), lr=actor_lr)
        self.Critic_optimizer = optim.Adam(self.Critic.parameters(), lr=critic_lr)
        self.Critic_loss = nn.MSELoss()


    def run(self, state):
        state = Variable(torch.from_numpy(state.copy()).float()).to(self.device)
        action = self.Actor.forward(state)
        return action.cpu().detach().numpy()  # if cuda is not available .cpu() is not necessary

    def train(self):

        states, actions, rewards, next_states, _ = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        # Prepare Actor Update
        self.Actor.to(self.device)
        Actor_actions = self.Actor.forward(states)
        Q = self.Critic.forward(states, Actor_actions)
        Actor_update = - Q.mean()

        # Prepare Critic Update
        Q = self.Critic.forward(states, actions)
        Actor_next_actions = self.Target_Actor.forward(next_states)
        Qnext = self.Target_Critic.forward(next_states, Actor_next_actions)
        Critic_update = self.Critic_loss(Q, Qnext * self.discount + rewards)

        # Update Actor NN
        self.Actor_optimizer.zero_grad()
        Actor_update.backward()
        self.Actor_optimizer.step()

        # Update Critic NN
        self.Critic_optimizer.zero_grad()
        Critic_update.backward()
        self.Critic_optimizer.step()

        # Update Target Networks
        for target_param, param in zip(self.Target_Actor.parameters(), self.Actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.Target_Critic.parameters(), self.Critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

import matplotlib.pyplot as plt

# env = gym.make('LunarLanderContinuous-v2')
# env = gym.make('Pendulum-v0')
env = secretary_task_env()
env.reset()
batch_size = 64
Agent = Ddpg(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], actor_hid=[256, 128],
               critic_hid=[256, 128], actor_lr=0.002, critic_lr=0.001,
               gamma=0.99, tau=0.05, replay_size=100000, batch_size=batch_size)
Noise = OUNoise(env.action_space, min_sigma=0)
instan_reward = []
cumulative_reward = []
Noise.reset()
getting_diamond = []
getting_high_ev = []
time_spent = []
visited = []
dist_1 = []
dist_2 = []
dist_3 = []
dist_4 = []
for episode in range(100):
    observation = env.reset()  # observation = state
    Noise.reset()

    rewards = 0

    for steps in range(500):
        action = Agent.run(observation)
        action = Noise.get_action(action, steps*episode)
        next_observation, reward, done, _ = env.step(action)  # take a random action
        rewards += reward
        # env.render()  # To see the agent while training uncomment this

        Agent.memory.push(observation, action, reward, next_observation, done)
        if len(Agent.memory) > batch_size:
            Agent.train()

        observation = next_observation
        dist_1.append(next_observation[2])
        dist_2.append(next_observation[3])
        dist_3.append(next_observation[4])
        dist_4.append(next_observation[5])

        if done:
            getting_diamond.append(env.getting_diamond)
            getting_high_ev.append(env.choosing_highest_ev)
            time_spent.append(env.total_time)
            instan_reward.append(rewards)
            cumulative_reward.append(np.mean(instan_reward[-10:]))
            print("Episode " + str(episode) + " has been completed with:")
            print("Reward: " + str(instan_reward[len(instan_reward) - 1]))
            print("Average Cumulative Reward: " + str(cumulative_reward[len(cumulative_reward) - 1]))
            print("At Step: " + str(steps))
            print("Total time: " + str(env.total_time))
            print()

            break
        elif steps == 499:
            instan_reward.append(rewards)
            cumulative_reward.append(np.mean(instan_reward[-10:]))
            getting_diamond.append(env.getting_diamond)
            getting_high_ev.append(env.choosing_highest_ev)
            time_spent.append(env.total_time)

    visited.append(env.visited)

    if rewards >= max(instan_reward):
        try:
            torch.save(Agent.Actor.state_dict(), "BestActor.pth")
            torch.save(Agent.Critic.state_dict(), "BestCritic.pth")
            torch.save(Agent.Target_Actor.state_dict(), "BestTargetActor.pth")
            torch.save(Agent.Target_Critic.state_dict(), "BestTargetCritic.pth")
        except:
            print("Bests could not saved")
            pass

for i in range(len(getting_diamond)):
    if getting_diamond[i]:
        plt.axvline(x=(i+1)*500, color="black", label="got diamond")
plt.title("steps vs. distances")
plt.xlabel('steps - every 500 = 1 episode')
plt.ylabel('distances')
plt.plot(np.arange(len(dist_1)), dist_1, label="distance_one",color="red")
plt.plot(np.arange(len(dist_1)), dist_2, label="distance_two",color="blue")
plt.plot(np.arange(len(dist_1)), dist_3, label="distance_three",color="green")
plt.plot(np.arange(len(dist_1)), dist_4, label="distance_four",color="orange")
plt.legend()
plt.show()
plt.title("Rewards vs Episodes")
poly = np.polyfit(np.arange(len(instan_reward)),instan_reward,10)
poly_y = np.poly1d(poly)(np.arange(len(instan_reward)))
plt.plot(np.arange(len(instan_reward)),poly_y, label="smoothed instant reward")
# plt.plot(np.arange(10),instan_reward, label="original")
plt.plot(instan_reward, label="Instant Reward", c="pink")
poly = np.polyfit(np.arange(len(cumulative_reward)), cumulative_reward,10)
poly_y = np.poly1d(poly)(np.arange(len(cumulative_reward)))
plt.plot(np.arange(len(cumulative_reward)),poly_y, label="smoothed last 10 reward", c="black")
plt.plot(cumulative_reward, label="Last 10 Reward Mean")
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend(loc='best')
plt.savefig("Figure.jpeg")
plt.show()

print("Number of times that got diamond: " + str(np.count_nonzero(getting_diamond)))
print("NUmber of times getting highest ev: " + str(np.count_nonzero(getting_high_ev)))

plt.plot(np.arange(len(visited)),visited)
plt.xlabel('Episodes')
plt.ylabel('# of times that diamonds are visited')
plt.show()

plt.plot(np.arange(len(time_spent)),time_spent)
plt.xlabel('episodes')
plt.ylabel('total time spent to get the diamond')
plt.show()

env.close()
