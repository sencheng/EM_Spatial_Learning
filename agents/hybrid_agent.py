import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.compat.v1.keras import callbacks, backend
from rl_adapted.util import *
from rl_adapted.core import Agent
from agents.dqn_adapted import DQNAgentAdapted
from agents.em_control import EM_control


'''
This agent is a hybrid agent between DQN and Episodic control agent for simulating the learning
paradigm of a healthy rat's brain.
'''

class HybridAgent(Agent):
    ### The nested visualization class that is required by 'KERAS-RL' to visualize the training success (by means of episode reward)
    ### at the end of each episode, and update the policy visualization.
    class callbacks(callbacks.Callback):
        # rlParent:     the ACT_ReinforcementLearningModule that hosts this class
        # trialBeginFcn:the callback function called in the beginning of each trial, defined for more flexibility in scenario control
        # trialEndFcn:  the callback function called at the end of each trial, defined for more flexibility in scenario control
        def __init__(self, rlParent, trialBeginFcn=None, trialEndFcn=None):

            super(HybridAgent.callbacks, self).__init__()

            # store the hosting class
            self.rlParent = rlParent

            # store the trial end callback function
            self.trialBeginFcn = trialBeginFcn

            # store the trial end callback function
            self.trialEndFcn = trialEndFcn

            # store the logs data
            self.logs_data=[]

        # The following function is called whenever an epsisode starts,
        # and updates the visual output in the plotted reward graphs.
        def on_episode_begin(self, epoch, logs):

            # retrieve the Open AI Gym interface
            interfaceOAI = self.rlParent.interfaceOAI

            if self.trialBeginFcn is not None:
                self.trialBeginFcn(epoch, self.rlParent)

        # The following function is called whenever an episode ends, and updates the reward accumulator,
        # simultaneously updating the visualization of the reward function
        def on_episode_end(self, epoch, logs):
            # store the log data
            self.logs_data.append(logs)
            if self.trialEndFcn is not None:
                self.trialEndFcn(epoch, self.rlParent, logs)

    def __init__(self, interfaceOAI, memoryCapacity=10000, epsilon=0.1, hybrid_format='weighted_sum', sigma=0.5, gamma=0.9, with_replay=True,
                trialBeginFcn=None, trialEndFcn=None):
        '''
        A combination of a CNN-DQN and model-free episodic control
        Parameters:
            hybrid_format: how the Q values from EC and DQN are combined. Options: 'weighted_sum' or 'take_max'
            sigma: the weight for summing the Q values, only needed when hybrid_format='weighted_sum'
        '''
        super().__init__()

        self.interfaceOAI = interfaceOAI
        self.semantic_agent = DQNAgentAdapted(interfaceOAI=interfaceOAI, enable_CNN=True, memory_type='sequential', batch_size=32, gamma=gamma,
                                        memoryCapacity=memoryCapacity, allow_replay=with_replay)
        self.episodic_agent = EM_control(interfaceOAI=interfaceOAI, epsilon=epsilon, memoryCapacity=memoryCapacity, gamma=gamma)
        self.epsilon = epsilon

        self.hybrid_format = hybrid_format
        self.sigma = sigma

        self.engagedCallbacks = self.callbacks(self, trialBeginFcn, trialEndFcn)
        # the number of discrete actions, retrieved from the Open AI Gym interface
        self.nb_actions = self.interfaceOAI.action_space.n # normally 3 or 6

        self.compiled = True

        # for recording the source of selcted Q values
        self.select_history = []
        self.episode_selection = []

    def forward(self, observation):
        # Syncroize the self.step parameter of the hybrid agent with the two sub-agents
        self.episodic_agent.step = self.step
        self.semantic_agent.step = self.step
        # retrieve Q values from EC
        Q_EC = self.episodic_agent.compute_q_value(observation)
        # infer Q values from Deep Q neural network
        Q_NN = self.semantic_agent.compute_q_value(observation)
        # combine Q values from both agents
        if self.hybrid_format == 'weighted_sum':
            Q_values = self.sigma*Q_EC + (1-self.sigma)*Q_NN
        elif self.hybrid_format == 'take_max':
            Q_values = np.asarray([max(x, y) for (x, y) in zip(Q_EC, Q_NN)])
        # action selection
        seed=np.random.random()
        if not self.training:  # if we are not in training phase but test phase, no exploration
            seed=1
        if seed<self.epsilon:
            action=np.random.randint(low=0, high=self.nb_actions)
        else:
            best_actions=np.argwhere(Q_values==np.max(Q_values)).flatten()
            if len(best_actions)==1:
                action=best_actions[0]
            else:
                action=np.random.choice(best_actions)
            #### At the same time, record if the selected Q value is from EC or DQN ###
            max_q_idx = np.argmax(np.concatenate((Q_EC, Q_NN)))
            if max_q_idx < self.nb_actions:
                self.episode_selection.append('EC')
            else:
                self.episode_selection.append('NN')

        # record the observation and the action
        self.episodic_agent.book_keeping(observation, action)
        self.semantic_agent.book_keeping(observation, action)

        return action

    def backward(self, reward, terminal):
        self.episodic_agent.backward(reward, terminal)
        metrics = self.semantic_agent.backward(reward, terminal)
        if terminal:
            self.select_history.append(self.episode_selection)
            self.episode_selection = []
        return metrics

    ### The following function is called to train the agent.
    def train(self, total_episodes, max_episode_steps=1000):
        # activate the self.training parameter in the two sub-agents
        self.episodic_agent.training = True
        self.semantic_agent.training = True
        # call the fit method to start the RL learning process
        history = self.fit(self.interfaceOAI, nb_episodes=total_episodes, verbose=2, callbacks=[self.engagedCallbacks],
                 nb_max_episode_steps=max_episode_steps, visualize=False)
        return history

    def predict(self, total_episodes, max_episode_steps=1000):
        # deactivate the self.training parameter in the two sub-agents
        self.episodic_agent.training = False
        self.semantic_agent.training = False
        history_data=self.test(self.interfaceOAI, nb_episodes=total_episodes, callbacks=[self.engagedCallbacks],
                  visualize=False, nb_max_episode_steps=max_episode_steps, verbose=2)
        return history_data
