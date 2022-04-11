from tensorflow.compat.v1.keras import backend, callbacks
from rl_adapted.core import Agent, Processor
from tensorflow.keras.models import Sequential, Model
from sklearn.neighbors import KDTree
import numpy as np
from collections import deque
import random
import time
from decimal import *


class Q_ec:
    '''
    This is the Q_ec table from the paper. There are three kinds of elements in this class: states, q_values and the most
    recent time the state has been visited.
        States: implemented as a KD-Tree for fast query of the index of the nearst point from the current state, the KD-Tree is
                constantly reconstructed.
        Q_values: implemented as a normal list
        last_visited_times: implemented as a normal list
    '''
    def __init__(self, max_size, nb_action, decay_rate):
        self.states=[]
        self.Q_values=[]
        self.last_visited_times=[]
        self.states_tree=None # since KD tree must be built upon some data, so initilized as None
        self.max_size=max_size # the length of the memory
        self.nb_action=nb_action
        self.decay_rate = decay_rate

    def get_length(self):
        return len(self.states)

    def estimate(self, state, k): # estimate Q values by using a kd-tree
        '''
        Estimate all the Q values of the current state with k nearst neighbors
        '''
        # print(len(self.states))
        if self.states_tree and len(self.states)>=k:
            nearst_idx=self.states_tree.query(X=[state], return_distance=False)[0][0]
            # print(np.linalg.norm(state-self.states[nearst_idx]))
            if np.linalg.norm(state-self.states[nearst_idx])<1.0:
                # if the nearst point is close enough to the quey point
                q_values = self.Q_values[nearst_idx]
                return q_values
            else:
                q_values=np.zeros(self.nb_action)
                # # for continous state space, find the k nearst indices
                k_nearst_idx=self.states_tree.query(X=[state], k=k, return_distance=False)[0]
                for idx in k_nearst_idx:
                    temp = self.Q_values[idx]
                    q_values = q_values + temp
                q_values/=k
                # for discrete space, there's no reason to estimate from other states, since each state is quite different
                # so we let the agent just randomly choose a value, so return all-zeros Q
                return q_values
        else:
            return np.zeros(self.nb_action)

    def update(self, episode):
        for event in episode:
            state=event['state']
            action=event['action']
            accmu_reward=event['accumulative']
            visited_time=event['last_visited']
            self.add_event(state, action, accmu_reward, visited_time)


    def add_event(self, state, action, accmu_reward, visited_time):
        # add event by using a kd-tree
        if self.states_tree:
            nearst_idx=self.states_tree.query(X=[state], return_distance=False)[0][0]
            # print(np.linalg.norm(state-self.states[nearst_idx]))
            if np.linalg.norm(state-self.states[nearst_idx])<1.0:
                self.Q_values[nearst_idx][action]=max(accmu_reward, self.Q_values[nearst_idx][action])
                self.last_visited_times[nearst_idx]=visited_time
            else:
                self.add(state, action, accmu_reward, visited_time)
        else:
            self.add(state, action, accmu_reward, visited_time)

    def add(self, state, action, accmu_reward, visited_time):
        if len(self.states)<self.max_size:
            self.states.append(state)
            new_q_values=np.zeros(self.nb_action)
            new_q_values[action]=accmu_reward
            self.Q_values.append(new_q_values)
            self.last_visited_times.append(visited_time)
        else:
            min_time_idx=np.argmin(self.last_visited_times)
            self.states[min_time_idx]=state
            new_q_values=np.zeros(self.nb_action)
            new_q_values[action]=accmu_reward
            self.Q_values[min_time_idx]=new_q_values
            self.last_visited_times[min_time_idx]=visited_time
        self.states_tree=KDTree(np.asarray(self.states))

    def decay(self):
        '''
        At each time step, the Q values naturally decay under a certain rate
        '''
        for i in range(len(self.Q_values)):
            self.Q_values[i] *= (1-self.decay_rate)



class EM_control(Agent):
    ### The nested visualization class that is required by 'KERAS-RL' to visualize the training success (by means of episode reward)
    ### at the end of each episode, and update the policy visualization.
    class callbacks(callbacks.Callback):

        # The constructor.
        #
        # rlParent:     the ACT_ReinforcementLearningModule that hosts this class
        # trialBeginFcn:the callback function called in the beginning of each trial, defined for more flexibility in scenario control
        # trialEndFcn:  the callback function called at the end of each trial, defined for more flexibility in scenario control
        def __init__(self, rlParent, trialBeginFcn=None, trialEndFcn=None):

            super(EM_control.callbacks, self).__init__()

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


    def __init__(self, interfaceOAI, memoryCapacity=2000, k=3, gamma=0.9, epsilon=0.1, processor=None, decay_rate=0.0,
                 use_random_projections=True, pretrained_model=None, trialBeginFcn=None, trialEndFcn=None):
        '''
        params:
            interfaceOAI: openAI interface
            memoryCapacity: the maximal length of the memory
            k: the k-nearest method when estimating the Q function for a state-action pair
            gamma: discount factor for the rewards in rl
            epsilon: episilon greedy method in Q learning framework
            Processor: required for the Agent parent class
        others:
            episodic_memory: the element inside EM is a dict() which stores a state, its Q values on each action, and when
                             the state was visited last time
        '''

        super().__init__(processor=processor)

        # store the Open AI Gym interface
        self.interfaceOAI = interfaceOAI

        self.memoryCapacity=memoryCapacity
        # the number of discrete actions, retrieved from the Open AI Gym interface
        self.nb_action = self.interfaceOAI.action_space.n
        self.k=k
        self.gamma=gamma
        self.epsilon=epsilon

        self.episodic_memory=Q_ec(self.memoryCapacity, self.nb_action, decay_rate)
        self.current_episode=deque()
        self.current_state=None
        self.current_action=0
        self.use_random_projections=use_random_projections
        self.init_projection=False
        self.pretrained_model=pretrained_model
        self.state_size=256  # size of the embedded state used by random projections

        # set up the visualizer for the RL agent behavior/reward outcome
        self.engagedCallbacks = self.callbacks(self, trialBeginFcn, trialEndFcn)
        # compile the model
        self.compile()


    def forward(self, observation):
        '''
        Perform epsilon-greedy to select an action, update the current state and action
        '''
        # process the observation and embed it to a lower-dimension
        current_state = self.process_observation(observation)
        # extract the Q values for this state from EM
        action_values=self.episodic_memory.estimate(current_state, self.k)
 
        seed=np.random.random()
        if not self.training:  # if we are not in training phase but test phase, no exploration
            seed=1
        if seed<self.epsilon:
            action_idx=np.random.randint(low=0, high=self.nb_action)
        else:
            best_actions=np.argwhere(action_values==np.max(action_values)).flatten()
            if len(best_actions)==1:
                action_idx=best_actions[0]
            else:
                action_idx=np.random.choice(best_actions)
        self.recent_observation=current_state
        self.recent_action=action_idx

        return action_idx

    def book_keeping(self, observation, action):
        '''
        Record the most recent obs and action
        '''
        # process the observation and embed it to a lower-dimension
        current_state = self.process_observation(observation)
        self.recent_observation = current_state
        self.recent_action = action

    def backward(self, reward, terminal):
        '''
        The agent observe the reward and a terminal signal. If the episode is not terminal, just store the current state, action, reward tuple
        in the self.current_episode; if terminate, update the episodic memory with the current episode
        '''
        event={
            'state': self.recent_observation,
            'action': self.recent_action,
            'reward': reward,
            'last_visited': time.time()
        }
        # store the current event
        self.current_episode.append(event)
        # if the episode terminates, update the episodic memory
        if terminal:
            self.current_episode=self.accmu_reward(self.current_episode)
            self.episodic_memory.update(self.current_episode)
            self.current_episode.clear()
            # decay the Q values
            self.episodic_memory.decay()

        # Only for coding compatibility
        metrics = [0.0]
        return metrics

    def accmu_reward(self, episode):
        episode_length = len(episode)
        for i in range(episode_length):
            accumulative=0.0
            for j in range(episode_length-1, i-1, -1):     # from the end step to the ith step
                accumulative=(accumulative+episode[j]['reward'])*self.gamma
            accumulative/=self.gamma  # divide the extra gamma
            episode[i]['accumulative']=accumulative
        return episode


    def compile(self):
        '''
        Required by the Agent parent class, here we don't have neural network model, so nothing is to be done, only setup the parameter
        '''
        self.compiled=True


    def process_observation(self, observation):
        '''
        The function for mapping the original, high-dimensional observation data to more abstract, lower-dimensional state.
        The methods can be random projection or by using a pretrained model like Variational autoencoder (VAE).
        '''
        if self.pretrained_model!=None:
            observation=self.pretrained_model.get_activations(observation).flatten()

        if self.use_random_projections:
            # flatten the observation into a single vector
            observation=observation.flatten()
            # if for the first time, generate a random projection
            if not self.init_projection:
                dim_h=len(observation)
                dim_low=self.state_size
                self.random_projections=np.float32(np.random.randn(dim_h, dim_low))
                self.init_projection=True
            processed_observation=np.dot(observation, self.random_projections)

        return processed_observation

    def compute_q_value(self, observation):
        processed_state = self.process_observation(observation)
        # extract the Q values for this state from EM
        q_value=self.episodic_memory.estimate(processed_state, self.k)
        return q_value

    ### The following function is called to train the agent.
    def train(self, total_episodes, max_episode_steps=1000, lr=0.00015):
        # call the fit method to start the RL learning process
        self.fit(self.interfaceOAI, nb_episodes=total_episodes, verbose=2, callbacks=[self.engagedCallbacks],
                       nb_max_episode_steps=max_episode_steps, visualize=False)
      

    def predict(self, total_episodes, max_episode_steps=1000):
        history_data=self.test(self.interfaceOAI, nb_episodes=total_episodes, callbacks=[self.engagedCallbacks],
                  visualize=False, nb_max_episode_steps=max_episode_steps, verbose=2)
        return history_data
