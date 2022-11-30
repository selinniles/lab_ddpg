import gym
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import base64, io
from gym import spaces
from gym.utils import seeding
import numpy as np
from collections import deque, namedtuple
import sympy

from os import path
from typing import Optional


class secretary_task_env(gym.Env):

    def __init__(self):
        self.num_diamonds = 4
        # coordinates for the agent and the diamonds
        self.agent_coordinate = np.zeros(2)
        nump_range = np.concatenate([np.arange(-10, 0, dtype=np.int), np.arange(1, 11, dtype=np.int)])
        x_pts = np.random.choice(nump_range, self.num_diamonds)
        y_pts = np.random.choice(nump_range,self.num_diamonds)
        self.diamond_coordinates = np.array([[x_pts[0], y_pts[0]], [x_pts[1], y_pts[1]], [x_pts[2], y_pts[2]], [x_pts[3], y_pts[3]]])
        self.dist_all = np.zeros(self.num_diamonds)
        self.all_evs = []

        self.choosing_time = 0
        self.visited = 0
        self.choosing_highest_ev = 0
        self.total_time = 0
        self.highest_ev = 0
        self.getting_diamond = 0
        self.chosen = False


        # probabilities and rewards
        self.probs = np.random.uniform(0.0, 1.0, self.num_diamonds)
        self.rewards = np.random.randint(0.0, 21.0, self.num_diamonds)
        for i in range(self.num_diamonds):
            self.all_evs.append(self.rewards[i]*self.probs[i])

        # self.action_space = > BOX = > [theta, velocity]
        h = np.array([
            np.pi,5
        ], dtype=np.float32)
        l = np.array([
            -np.pi,0
        ], dtype=np.float32)
        self.action_space = spaces.Box(l, h, dtype=np.float32)

        #observation space = > BOX => [pos_x, pos_y, d1, d2, d3, d4, total_time, last_seen_ev]
        high = np.array([
            10,10,
            29,29,29,29,
            500,
        ],dtype=np.float32)
        low = np.array([
            -10,-10,
            0,0,0,0,
            0,
        ],dtype=np.float32)

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()

        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg

        # get the action
        # action[0] = np.clip(action[0], -np.pi, np.pi)
        # action[1] = np.clip(action[1], 0, 5)
        action= action[0]
        reward = 0
        done = False
        theta = action[0]
        velocity = action[1]
        self.total_time += 1
        if velocity == 0:
            self.choosing_time += 1
        else:
            self.choosing_time = 0

        pos_x, pos_y, d1, d2, d3, d4, time_spent = self.state
        diamond_index = None

        r = velocity
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pos_x += x
        pos_y += y

        time_spent = self.total_time

        pos_x = np.clip(pos_x, -10, 10).astype(np.float32)
        pos_y = np.clip(pos_y, -10, 10).astype(np.float32)

        self.agent_coordinate = pos_x, pos_y

        d1 = np.linalg.norm(self.diamond_coordinates[0] - self.agent_coordinate).astype(np.float32)
        d2 = np.linalg.norm(self.diamond_coordinates[1] - self.agent_coordinate).astype(np.float32)
        d3 = np.linalg.norm(self.diamond_coordinates[2] - self.agent_coordinate).astype(np.float32)
        d4 = np.linalg.norm(self.diamond_coordinates[3] - self.agent_coordinate).astype(np.float32)

        self.dist_all = d1, d2, d3, d4

        for i in range(len(self.dist_all)):
            if self.dist_all[i] < 1:
                diamond_index = i
                self.visited += 1
            if diamond_index is not None:
                new_ev = self.rewards[diamond_index] * self.probs[diamond_index]
                if self.choosing_time >= 3:
                    print("got the diamond")
                    done = True
                    self.getting_diamond = 1
                    if new_ev >= self.highest_ev:
                        self.highest_ev = new_ev
                    reward_occur = np.random.binomial(1, self.probs[diamond_index])
                    if reward_occur:
                        reward += self.rewards[diamond_index]
                    else:
                        reward += 0

                    if self.highest_ev == np.max(np.array(self.all_evs)):
                        print("Got Highest EV!")
                        self.choosing_highest_ev = 1
                    break

        reward -= 0.1
        self.state = np.array([pos_x, pos_y, d1, d2, d3, d4, time_spent]).astype(np.float32)
        return self.state, reward, done, {}

    def reset(self):
        self.all_evs = []
        self.visited = 0
        self.probs = np.random.uniform(0.0, 1.0, self.num_diamonds)
        self.rewards = np.random.randint(0.0, 21.0, self.num_diamonds)
        nump_range = np.concatenate([np.arange(-10, 0, dtype=np.int), np.arange(1, 11, dtype=np.int)])
        x_pts = np.random.choice(nump_range, self.num_diamonds)
        y_pts = np.random.choice(nump_range, self.num_diamonds)
        self.agent_coordinate = np.zeros(2)
        self.diamond_coordinates = [[x_pts[0], y_pts[0]], [x_pts[1], y_pts[1]], [x_pts[2], y_pts[2]],
                                    [x_pts[3], y_pts[3]]]
        self.diamond_coordinates = np.array(self.diamond_coordinates)
        self.choosing_time = 0
        self.total_time = 0
        self.highest_ev = 0
        self.chosen = False
        self.choosing_highest_ev = 0
        self.getting_diamond = 0

        for i in range(self.num_diamonds):
            self.all_evs.append(self.rewards[i]*self.probs[i])

        d1 = np.linalg.norm(self.diamond_coordinates[0] - self.agent_coordinate)
        d2 = np.linalg.norm(self.diamond_coordinates[1] - self.agent_coordinate)
        d3 = np.linalg.norm(self.diamond_coordinates[2] - self.agent_coordinate)
        d4 = np.linalg.norm(self.diamond_coordinates[3] - self.agent_coordinate)

        self.state = np.array([self.agent_coordinate[0], self.agent_coordinate[1], d1, d2, d3, d4, self.total_time])

        return self.state
