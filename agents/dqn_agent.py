# basic imports
import numpy as np
from collections import deque, namedtuple
# keras imports
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
from memory_modules.memories import NoMemory, SequentialMemory, SparseSequentialMemory
# keras-rl imports
from agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy
from rl.core import Processor

class CustomProcessor(Processor):
    '''
    acts as a coupling mechanism between the agent and the environment
    '''

    def process_state_batch(self, batch):
        '''
        Given a state batch, I want to remove the second dimension, because it's
        useless and prevents me from feeding the tensor into my CNN
        '''
        return np.squeeze(batch, axis=1)



class DQNAgentBaseline():
    '''
    This class implements a DQN agent.
    The original step-based training behavior of the keras-rl2 DQN agent is overwritten to be trial-based.

    | **Args**
    | interfaceOAI:                 The interface to the Open AI Gym environment
    | memoryCapacity:               The capacity of the sequential memory used in the agent.
    | epsilon:                      The epsilon value for the epsilon greedy policy.
    | trialBeginFcn:                The callback function called at the beginning of each trial, defined for more flexibility in scenario control.
    | trialEndFcn:                  The callback function called at the end of each trial, defined for more flexibility in scenario control.
    | model:                        The network model to be used by the DQN agent. If None, a default network will be created.
    '''

    class callbacks(callbacks.Callback):
        '''
        Callback class. Used for visualization and scenario control.

        | **Args**
        | rlParent:                     Reference to the RL agent.
        | trialBeginFcn:                The callback function called at the beginning of each trial, defined for more flexibility in scenario control.
        | trialEndFcn:                  The callback function called at the end of each trial, defined for more flexibility in scenario control.
        '''

        def __init__(self, rlParent, trialBeginFcn=None, trialEndFcn=None):
            # store the hosting class
            self.rlParent = rlParent
            # store the trial end callback function
            self.trialBeginFcn = trialBeginFcn
            # store the trial end callback function
            self.trialEndFcn = trialEndFcn

        def on_episode_begin(self, epoch, logs):
            '''
            The following function is called whenever an epsisode starts.

            | **Args**
            | epoch:                        The current trial.
            | logs:                         A dictionary containing logs of the simulation.
            '''
            if self.trialBeginFcn is not None:
                self.trialBeginFcn(self.rlParent.current_trial - 1, self.rlParent)

        def on_episode_end(self, epoch, logs):
            '''
            The following function is called whenever an episode ends, and updates the reward accumulator,
            simultaneously updating the visualization of the reward function.

            | **Args**
            | epoch:                        The current trial.
            | logs:                         A dictionary containing logs of the simulation.
            '''
            # update trial count
            self.rlParent.current_trial += 1
            self.rlParent.session_trial += 1
            # stop training after the maximum number of trials was reached
            if self.rlParent.session_trial >= self.rlParent.max_trials:
                self.rlParent.agent.step = self.rlParent.max_steps + 1
            if self.trialEndFcn is not None:
                self.trialEndFcn(self.rlParent.current_trial - 1, self.rlParent, logs)

    def __init__(self, modules, memoryCapacity=1000000, epsilon=0.3, batch_size=32, lr=0.0001, sparse_sample=False,
                 trialBeginFcn=None, trialEndFcn=None, model=None):
        # store the Open AI Gym interface
        self.interfaceOAI = modules['rl_interface']
        # retrieve number of actions
        self.nb_actions = self.interfaceOAI.action_space.n
        # and also the observation space
        self.observation_space = self.interfaceOAI.observation_space
        # build model
        self.model = model
        if self.model is None:
            self.build_model()
        # prepare the memory for the RL agent
        if not sparse_sample:
            self.memory = SequentialMemory(limit=memoryCapacity, window_length=1)
        else:
            self.memory = SparseSequentialMemory(sample_interval=batch_size, limit=memoryCapacity, window_length=1)

        if memoryCapacity == 0:
            self.memory = NoMemory(window_length=1)
        # define the available policies
        policy = EpsGreedyQPolicy(epsilon)
        # define the maximum number of steps
        self.max_steps = 10 ** 10
        # keeps track of current trial
        self.current_trial = 0  # trial count across all sessions (i.e. calls to the train/simulate method)
        self.session_trial = 0  # trial count in current seesion (i.e. current call to the train/simulate method)
        # define the maximum number of trials
        self.max_trials = 0
        # construct the agent
        self.agent = DQNAgent(model=self.model, nb_actions=self.nb_actions, memory=self.memory, gamma=0.95,
                              nb_steps_warmup=batch_size, enable_dueling_network=False, enable_double_dqn=False, processor=None,
                              dueling_type='avg', target_model_update=1e-2, policy=policy, batch_size=batch_size)
        # compile the agent
        self.agent.compile(Adam(lr=lr, ), metrics=['mse'])
        # set up the visualizer for the RL agent behavior/reward outcome
        self.engagedCallbacks = self.callbacks(self, trialBeginFcn, trialEndFcn)

    def build_model(self):
        '''
        This function builds a default network model for the DQN agent.
        '''
        #### start defining the agent
        # we first define the NN model used by DQN
        # We use the same CNN that was described by Mnih et al. (2015)
        sensory_model = Sequential()
        sensory_model.add(Convolution2D(16, kernel_size=(8, 8), strides=4, activation='relu',
                                        input_shape=self.observation_space.shape))
        sensory_model.add(Convolution2D(32, kernel_size=(4, 4), strides=2, activation='relu'))
        sensory_model.add(Flatten())  # dimension: 3136
        feature_input = sensory_model.output
        x = Dense(256, activation='relu')(feature_input)
        x = Dense(self.nb_actions, activation='linear')(x)
        self.model = Model(inputs=sensory_model.input, outputs=x)

    def train(self, numberOfTrials=100, maxNumberOfSteps=100):
        '''
        This function is called to train the agent.

        | **Args**
        | numberOfTrials:               The number of trials the RL agent is trained.
        | maxNumberOfSteps:             The maximum number of steps per trial.
        | replayBatchSize:              The number of random that will be replayed.
        '''
        self.max_trials = numberOfTrials
        self.session_trial = 0
        # call the fit method to start the RL learning process
        self.agent.fit(self.interfaceOAI, nb_steps=self.max_steps, verbose=2, callbacks=[self.engagedCallbacks],
                       nb_max_episode_steps=maxNumberOfSteps, visualize=False)

    def predict(self, numberOfTrials=100, maxNumberOfSteps=100):
        self.max_trials = numberOfTrials
        self.session_trial = 0
        # call the predict method to start the RL learning process
        history_data = self.agent.test(self.interfaceOAI, nb_episodes=numberOfTrials, verbose=2,
                                 callbacks=[self.engagedCallbacks], nb_max_episode_steps=maxNumberOfSteps,
                                 visualize=False)
        return history_data

    def predict_on_batch(self, batch):
        '''
        This function retrieves Q-values for a batch of states.

        | **Args**
        | batch:                        The batch of states.
        '''
        return self.agent.model.predict_on_batch(batch)

    def reset_model(self):
        for ix, layer in enumerate(self.model.layers):
            if hasattr(self.model.layers[ix], 'kernel_initializer') and \
                    hasattr(self.model.layers[ix], 'bias_initializer'):
                weight_initializer = self.model.layers[ix].kernel_initializer
                bias_initializer = self.model.layers[ix].bias_initializer

                old_weights, old_biases = self.model.layers[ix].get_weights()

                self.model.layers[ix].set_weights([
                    weight_initializer(shape=old_weights.shape),
                    bias_initializer(shape=len(old_biases))])
