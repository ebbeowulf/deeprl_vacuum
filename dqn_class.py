"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import environment
import cv2
# import gym.spaces
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from pytorch_dqn.utils.replay_buffer import ReplayBuffer
# from utils.gym import get_wrapper_by_name

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": []
}

mean_episode_reward = -float('nan')
best_mean_episode_reward = -float('inf')
LOG_EVERY_N_STEPS = 500000

def save_stats(fName, env, t, learning_starts = 50000, exploration = None):
    global mean_episode_reward, best_mean_episode_reward, LOG_EVERY_N_STEPS
    episode_rewards = env.reward_history
    if len(episode_rewards) > 0:
        mean_episode_reward = np.mean(episode_rewards[-100:])
    if len(episode_rewards) > 100:
        best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

    Statistic["mean_episode_rewards"].append(mean_episode_reward)
    Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)

    if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
        print("Timestep %d" % (t,))
        print("mean reward (100 episodes) %f" % mean_episode_reward)
        print("best mean reward %f" % best_mean_episode_reward)
        print("episodes %d" % len(episode_rewards))
        if exploration is not None:
            print("exploration %f" % exploration.value(t))
        sys.stdout.flush()

        # Dump statistics to pickle
        with open(fName, 'wb') as f:
            pickle.dump(Statistic, f)
            print("Saved to %s" % fName)

class dqn:
    def __init__(self,
        env: environment.dirty_room,
        q_func,
        optimizer_spec,
        exploration,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        target_update_freq=10000
        ):
        self.env = env
        # Initialize target q function and q function
        self.input_arg = 3  # number of channels in input space
        self.num_actions = len(env.action_space)
        self.Q = q_func(self.input_arg, self.num_actions).type(dtype)
        self.target_Q = q_func(self.input_arg, self.num_actions).type(dtype)

        # Construct Q network optimizer function
        self.optimizer = optimizer_spec.constructor(self.Q.parameters(), **optimizer_spec.kwargs)

        # Construct the replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size, 1)

        self.exploration = exploration
        self.batch_size=batch_size
        self.gamma=gamma
        self.learning_starts=learning_starts
        self.learning_freq=learning_freq
        self.target_update_freq=target_update_freq
        self.DRAW_EVERY_X_GAMES = 100
        self.num_param_updates = 0
        self.draw_shape = (400, 400)
        self.obs_shape = (50, 50) #resize images to this for input to DQN
        self.train_t = 0

    # Construct an epsilson greedy policy with given exploration schedule
    def select_epsilson_greedy_action(self, model, obs, isTrain=True):
        sample = random.random()
        eps_threshold = self.exploration.value(self.train_t)
        if (not isTrain) or (sample > eps_threshold):
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
            # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            # return model(Variable(obs, volatile=True)).data.max(1)[1].cpu()
            with torch.no_grad(): 
                return model(Variable(obs)).data.max(1)[1].cpu()
        else:
            return torch.IntTensor([random.randrange(self.num_actions)])

    def play_game(self, draw_game=False, isTrain=True):
        obs_full = self.env.reset(num_solid_objects=np.random.choice([1, 2]))
        last_obs = cv2.resize(obs_full, self.obs_shape)
        for t in count():
            self.train_t += 1
            ### Step the env and store the transition
            # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
            last_idx = self.replay_buffer.store_frame(last_obs)
            # encode_recent_observation will take the latest observation
            # that you pushed into the buffer and compute the corresponding
            # input that should be given to a Q network by appending some
            # previous frames.
            recent_observations = self.replay_buffer.encode_recent_observation()

            # Choose random action if not yet start learning
            if self.train_t > self.learning_starts:
                # action = select_epilson_greedy_action(Q, recent_observations, t)[0, 0]
                # action = self.select_epsilson_greedy_action(self.Q, recent_observations, isTrain=isTrain)[0]
                action = self.select_epsilson_greedy_action(self.Q, recent_observations)[0]
            else:
                action = random.randrange(self.num_actions)
            # Advance one step
            # obs, reward, done, _ = env.step(action)
            obs_full, reward, done = self.env.act(action)
            obs = cv2.resize(obs_full, self.obs_shape)
            # clip rewards between -1 and 1
            reward = max(-1.0, min(reward, 1.0))
            # Store other info in replay memory
            self.replay_buffer.store_effect(last_idx, action, reward, done)

            ### 3. Log progress and keep track of statistics - needs to be done BEFORE the game reset
            ##      in order to capture end game rewards
            save_stats("dqn_statistics.pkl", self.env, self.train_t, self.learning_starts, self.exploration)

            if draw_game:
                # gt = self.env.draw_gt()
                # outI = np.hstack((obs,gt))
                # pdb.set_trace()
                outI2 = cv2.resize(obs_full, self.draw_shape)
                cv2.imshow('DQN Maze', outI2)
                cv2.waitKey(1)

            last_obs = obs

            ### Perform experience replay and train the network.
            # Note that this is only done if the replay buffer contains enough samples
            # for us to learn something useful -- until then, the model will not be
            # initialized and random actions should be taken
            if isTrain:
                if (self.train_t > self.learning_starts and
                        self.train_t % self.learning_freq == 0 and
                        self.replay_buffer.can_sample(self.batch_size)):
                    # Use the replay buffer to sample a batch of transitions
                    # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
                    # in which case there is no Q-value at the next state; at the end of an
                    # episode, only the current state reward contributes to the target
                    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.replay_buffer.sample(self.batch_size)
                    # Convert numpy nd_array to torch variables for calculation
                    obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype) / 255.0)
                    act_batch = Variable(torch.from_numpy(act_batch).long())
                    rew_batch = Variable(torch.from_numpy(rew_batch))
                    next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype) / 255.0)
                    not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)

                    if USE_CUDA:
                        act_batch = act_batch.cuda()
                        rew_batch = rew_batch.cuda()
                    # Compute current Q value, q_func takes only state and output value for every state-action pair
                    # We choose Q based on action taken.
                    current_Q_values = self.Q(obs_batch).gather(1, act_batch.unsqueeze(1))
                    # Compute next Q value based on which action gives max Q values
                    # Detach variable from the current graph since we don't want gradients for next Q to propagated
                    next_max_q = self.target_Q(next_obs_batch).detach().max(1)[0]
                    next_Q_values = not_done_mask * next_max_q
                    # Compute the target of the current Q values
                    target_Q_values = rew_batch + (self.gamma * next_Q_values)
                    # Compute Bellman error
                    bellman_error = target_Q_values - current_Q_values.reshape(target_Q_values.shape)
                    # clip the bellman error between [-1 , 1]
                    clipped_bellman_error = bellman_error.clamp(-1, 1)
                    # Note: clipped_bellman_delta * -1 will be right gradient
                    d_error = clipped_bellman_error * -1.0
                    # Clear previous gradients before backward pass
                    self.optimizer.zero_grad()
                    # run backward pass
                    current_Q_values.backward(d_error.data.unsqueeze(1))

                    # Perfom the update
                    self.optimizer.step()
                    self.num_param_updates += 1

                    # Periodically update the target network by Q network to target Q network
                    if self.num_param_updates % self.target_update_freq == 0:
                        self.target_Q.load_state_dict(self.Q.state_dict())
            if done:
                break

    def train(self, number_of_games, draw_every_x_games=100):
        ###############
        # RUN ENV     #
        ###############
        #DRAW_EVERY_X_GAMES = 100

        for game_count in range(number_of_games):
            if game_count % 10==0:
                print("Game: %d" % (game_count))
            draw_game=((game_count%draw_every_x_games)==0)
            if draw_game:
                print("Current Training Count: %d" % game_count)
            self.play_game(draw_game=draw_game,isTrain=True)
            self.env.reset(num_solid_objects=np.random.choice([1, 3, 5, 7]))

