#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import random

import numpy as np
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld


class DynaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))
        # TO DO: Initialize count tables, and reward sum tables.
        self.n_sas = np.zeros((n_states, n_actions, n_states))
        self.R_sum = np.zeros((n_states, n_actions, n_states))
        self.trans_prob = np.zeros((n_states, n_actions, n_states))
        self.avg_reward = np.zeros((n_states, n_actions, n_states))

    def select_action(self, s, epsilon):
        # e-greedy action selection
        a = np.argmax(self.Q_sa[s])
        if random.random() < epsilon:
            a = random.choice(range(self.n_actions))
        return a

    def update(self, s, a, r, done, s_next, n_planning_updates):
        # Dyna update

        # Updates
        self.n_sas[s, a, s_next] += 1
        self.R_sum[s, a, s_next] += r
        for state_index in range(self.n_states):
            self.trans_prob[s, a, state_index] = \
                self.n_sas[s, a, state_index] / np.sum(self.n_sas[s, a]) if np.sum(self.n_sas[s, a]) != 0 else 0
            self.avg_reward[s, a, state_index] = \
                self.R_sum[s, a, state_index] / self.n_sas[s, a, state_index] if self.n_sas[s, a, state_index] != 0 else 0

        # Update Q-values table
        self.Q_sa[s, a] = \
            self.Q_sa[s, a] + self.learning_rate * (r + self.gamma * np.max(self.Q_sa[s_next]) - self.Q_sa[s, a])

        # Update previous Q-table with previous steps
        for _ in range(n_planning_updates):
            # selection random previously observed state and action taken in that state
            mask = self.n_sas > 0
            indices = np.argwhere(mask)
            random_indices = indices[np.random.choice(indices.shape[0])]
            state, action = random_indices[:2]

            state_next = np.random.choice(range(self.n_states), p=self.trans_prob[state, action])
            reward = self.avg_reward[state, action, state_next]

            self.Q_sa[state, action] = \
                self.Q_sa[state, action] + \
                self.learning_rate * (reward + self.gamma * np.max(self.Q_sa[state_next]) - self.Q_sa[state, action])

    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s])  # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return


class PrioritizedSweepingAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, max_queue_size=200, priority_cutoff=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.priority_cutoff = priority_cutoff
        self.Q_sa = np.zeros((n_states, n_actions))
        # TO DO: Initialize count tables, reward sum tables, priority queue
        self.n_sas = np.zeros((n_states, n_actions, n_states))
        self.R_sum = np.zeros((n_states, n_actions, n_states))
        self.trans_prob = np.zeros((n_states, n_actions, n_states))
        self.avg_reward = np.zeros((n_states, n_actions, n_states))
        self.queue = PriorityQueue()

    def select_action(self, s, epsilon):
        # e-greedy action selection
        a = np.argmax(self.Q_sa[s])
        if random.random() < epsilon:
            a = random.choice(range(self.n_actions))
        return a

    def update(self, s, a, r, done, s_next, n_planning_updates):

        # TO DO: Add Prioritized Sweeping code

        # Helper code to work with the queue
        # Put (s,a) on the queue with priority p (needs a minus since the queue pops the smallest priority first)
        # self.queue.put((-p,(s,a))) 
        # Retrieve the top (s,a) from the queue
        # _,(s,a) = self.queue.get() # get the top (s,a) for the queue

        # Updates
        self.n_sas[s, a, s_next] += 1
        self.R_sum[s, a, s_next] += r
        for state_index in range(self.n_states):
            self.trans_prob[s, a, state_index] = \
                self.n_sas[s, a, state_index] / np.sum(self.n_sas[s, a]) if np.sum(self.n_sas[s, a]) != 0 else 0
            self.avg_reward[s, a, state_index] = \
                self.R_sum[s, a, state_index] / self.n_sas[s, a, state_index] if self.n_sas[s, a, state_index] != 0 else 0

        p = abs(r + self.gamma * np.max(self.Q_sa[s_next]) - self.Q_sa[s, a])
        if p > self.priority_cutoff:
            self.queue.put((-p, (s, a)))

        # Update Q-table with the states from the queue
        for _ in range(n_planning_updates):
            if self.queue.empty():
                break
            _, (state, action) = self.queue.get()
            state_next = np.random.choice(range(self.n_states), p=self.trans_prob[state, action])
            reward = self.avg_reward[state, action, state_next]
            self.Q_sa[state, action] = \
                self.Q_sa[state, action] + \
                self.learning_rate * (reward + self.gamma * np.max(self.Q_sa[state_next]) - self.Q_sa[state, action])

            # Get over all actions, that may lead to state "state"
            prev_states_actions = self.trans_prob[:, :, state]
            mask = prev_states_actions > 0
            prev_state_action_pairs = np.argwhere(mask)

            for prev_state, prev_action in prev_state_action_pairs:
                r_bar = self.avg_reward[prev_state, prev_action, state]
                priority = abs(r_bar + self.gamma * np.max(self.Q_sa[state]) - self.Q_sa[prev_state, prev_action])

                # Add the previous pair into the priority queue if priority exceeds the treshold
                if priority > self.priority_cutoff:
                    self.queue.put((-priority, (prev_state, prev_action)))

    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s])  # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return


def ttest():
    n_timesteps = 101
    gamma = 1.0

    # Algorithm parameters
    policy = 'ps'  # or 'dyna'
    epsilon = 0.1
    learning_rate = 0.2
    n_planning_updates = 3

    # Plotting parameters
    plot = True
    plot_optimal_policy = True
    step_pause = 0.000001

    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states, env.n_actions, learning_rate, gamma)  # Initialize Dyna policy
    elif policy == 'ps':
        pi = PrioritizedSweepingAgent(env.n_states, env.n_actions, learning_rate, gamma)  # Initialize PS policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))

    # Prepare for running
    s = env.reset()
    continuous_mode = False

    for t in range(n_timesteps):
        # Select action, transition, update policy
        a = pi.select_action(s, epsilon)
        s_next, r, done = env.step(a)
        pi.update(s=s, a=a, r=r, done=done, s_next=s_next, n_planning_updates=n_planning_updates)

        # Render environment
        if plot:
            env.render(Q_sa=pi.Q_sa, plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)

        # Ask user for manual or continuous execution
        if not continuous_mode:
            key_input = input("Press 'Enter' to execute next step, press 'c' to run full algorithm")
            continuous_mode = True if key_input == 'c' else False

        # Reset environment when terminated
        if done:
            s = env.reset()
        else:
            s = s_next


if __name__ == '__main__':
    ttest()
