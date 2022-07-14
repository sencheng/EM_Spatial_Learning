import numpy as np
import tensorflow as tf
import os
from tensorflow.compat.v1.keras import callbacks, backend
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Convolution2D, SimpleRNN, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from rl_adapted.util import *
from rl_adapted.memory import SequentialMemory, NoMemory
from agents.semantic_dqn import DQNAgent
from rl_adapted.policy import EpsGreedyQPolicy
import time


class DQNAgentAdapted(DQNAgent):
    ### The nested visualization class that is required by 'KERAS-RL' to visualize the training success (by means of episode reward)
    ### at the end of each episode, and update the policy visualization.
    class callbacks(callbacks.Callback):
        '''
        # rlParent:     the ACT_ReinforcementLearningModule that hosts this class
        # trialBeginFcn:the callback function called in the beginning of each trial, defined for more flexibility in scenario control
        # trialEndFcn:  the callback function called at the end of each trial, defined for more flexibility in scenario control
        '''
        def __init__(self, rlParent, trialBeginFcn=None, trialEndFcn=None):

            super(DQNAgentAdapted.callbacks, self).__init__()

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

    def __init__(self, interfaceOAI, memoryCapacity=10000, epsilon=0.1, gamma=0.9, memory_type='sequential', total_steps=1000, n=5, allow_replay=True, lr=0.0001,
                 trialBeginFcn=None, trialEndFcn=None, enable_CNN=True, use_random_projections=False, batch_size=32, train_interval=1):

        # store the Open AI Gym interface
        self.interfaceOAI = interfaceOAI

        # the number of discrete actions, retrieved from the Open AI Gym interface
        self.nb_actions = self.interfaceOAI.action_space.n # normally 3 or 6

        # Get the list of observation shapes
        self.observation_space=self.interfaceOAI.observation_space # (84, 84, 3)

        self.use_random_projections=use_random_projections

        if self.use_random_projections:
            # if the size of the observation is too large, e.g. an image input, then compress it to a lower dim
            self.projection_space=256
            self.init_projection = False

        self.memory_type = memory_type

        ###### let the agent discover itself which model (FF, CNN, cross_modality) it should use #############
        if not enable_CNN:
            # we use random projection to reduce data dimension
            self.model = Sequential(
                [
                    Flatten(input_shape=(1, )+(self.projection_space,)),
                    Dense(units=256, activation='relu'),
                    Dense(units=128, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(units=self.nb_actions, activation='linear'),
                ]
            )
            self.custom_model_objects = {}
        else :
            # We use the same CNN that was described by Mnih et al. (2015)
            self.sensory_model=Sequential()
            self.sensory_model.add(Convolution2D(16, kernel_size=(8,8), strides=4, activation='relu', input_shape=self.observation_space.shape))
            self.sensory_model.add(Convolution2D(32, kernel_size=(4,4), strides=2, activation='relu'))
            # self.sensory_model.add(Convolution2D(32, kernel_size=(3,3), strides=1, activation='relu'))
            self.sensory_model.add(Flatten()) # dimension: 3136
            feature_input = self.sensory_model.output
            # x = Reshape((1, feature_input.shape[1]))(feature_input)
            # x = Dense(256, activation='relu')(x)
            x = Dense(256, activation='relu')(feature_input)
            x = Dense(self.nb_actions, activation='linear')(x)
            self.model = Model(inputs=self.sensory_model.input, outputs=x)
            self.custom_model_objects = {}

        # prepare the memory for the RL agent
        if memory_type=='prioritized':
            self.memory = PrioritizedMemory(limit=memoryCapacity, alpha=.8, start_beta=.4,
                                            end_beta=1., steps_annealed=int(total_steps*0.7), window_length=1)
            lr /= 2
        elif memory_type=='sequential':
            self.memory = SequentialMemory(limit=memoryCapacity, window_length=1)
        elif memory_type=='nstep':
            self.memory = NStepMemory(limit=n, window_length=1)

        # define the available policies
        policyEpsGreedy = EpsGreedyQPolicy(epsilon)
        # construct the DQN agent
        super(DQNAgentAdapted, self).__init__(model=self.model, nb_actions=self.nb_actions, memory=self.memory, with_replay=allow_replay, nb_steps_warmup=batch_size,
                                        enable_dueling_network=True, target_model_update=0.01, gamma=gamma, delta_clip=1., train_interval=train_interval,
                                        enable_double_dqn=True, policy=policyEpsGreedy, batch_size=batch_size, custom_model_objects=self.custom_model_objects)
        self.engagedCallbacks = self.callbacks(self, trialBeginFcn, trialEndFcn)

        # compile the agent
        self.compile(Adam(lr=lr), metrics=['mse'])

    def forward(self, observation):
        '''
        This function takes the image as input, compute the Q values of the input,
        and select an action based on the values.
        Parameters:
            Observation:             an RGB image or a 1-D vector
        Return: an action represented by a integer
        '''
        if self.use_random_projections:
            # compress the observation with random projection
            observation=[self.process_observation(observation)]
            action=super().forward(observation)
        else:
            action=super().forward(observation)
        action = np.int64(action)
        return action

    def book_keeping(self, observation, action):
        '''
        Record the most recent obs and action
        '''
        self.recent_observation = observation
        self.recent_action = action

    def change_replay(self, with_replay):
        '''
        Whether or not to use memory replay
        '''
        self.with_replay = with_replay

    def process_observation(self, observation):
        '''
        The function for mapping the original, high-dimensional observation data to more abstract, lower-dimensional state.
        The methods can be random projection or by using a pretrained model like Variational autoencoder (VAE).
        '''
        flattened_observation = observation.flatten()
        # if for the first time, generate a random projection
        if not self.init_projection:
            dim_h=len(flattened_observation)
            dim_low=self.projection_space
            self.random_projections=np.random.randn(dim_h, dim_low)
            self.init_projection = True
        observation=np.dot(flattened_observation, self.random_projections)
        return observation


    def train(self, total_episodes, max_episode_steps=1000):
        '''
        The function is called to train the agent.
        '''
        # call the fit method to start the RL learning process
        self.fit(self.interfaceOAI, nb_episodes=total_episodes, verbose=2, callbacks=[self.engagedCallbacks],
                 nb_max_episode_steps=max_episode_steps, visualize=False)


    def predict(self, total_episodes, max_episode_steps=1000):
        '''
            The function is called to test the agent's performance without
            training it.
        '''
        history_data=self.test(self.interfaceOAI, nb_episodes=total_episodes, callbacks=[self.engagedCallbacks],
                  visualize=False, nb_max_episode_steps=max_episode_steps, verbose=2)
        return history_data

    def save_model(self, weights_path, projection_path=None):
        '''
        Save the weights of the NN and the projection matrix, if random projection is
        used to pre-process the inputs
        '''
        self.model.save_weights(weights_path)
        if projection_path:
            pickle.dump(self.random_projections, open(projection_path, 'wb'))

    def load_model(self, weights_path, projection_path=None):
        '''
        # load the weights of nn and random projection values, if the latter were used
        '''
        self.model.load_weights(weights_path)
        if projection_path:
            self.init_projection = True
            self.random_projections = pickle.load(open(projection_path, 'rb'))

    def compute_q_value(self, observation):
        observation = np.array([observation])
        return self.model.predict(observation)[0]

    def sleep_phase(self, num_update):
        '''
        # perform offline replay after the training session
        '''
        for _ in range(num_update):
            metrics = self.replay()

    def reinit_layer(self, layer):
        '''
        This function takes in any NN layer, call its initilizer and reset
        the weights.
        '''
        session = backend.get_session()
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

    def reset_model(self):
        '''
        This function reset the weights of all layers
        '''
        for layer in self.model.layers:
            self.reinit_layer(layer)
