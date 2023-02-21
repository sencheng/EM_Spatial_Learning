# basic imports
import numpy as np
# keras imports
from tensorflow.compat.v1.keras import callbacks, backend
from tensorflow.keras.models import Sequential, Model, clone_model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
# keras-rl imports
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy
# custom imports
from memory_modules.memories import SequentialMemory_adapted

class SFMA_DQNAgent():
    class callbacks(callbacks.Callback):
        def __init__(self, rlParent, trialBeginFcn=None, trialEndFcn=None):
            # store the hosting class
            self.rlParent = rlParent
            # store the trial end callback function
            self.trialBeginFcn = trialBeginFcn
            # store the trial end callback function
            self.trialEndFcn = trialEndFcn

        def on_episode_begin(self, epoch, logs):
            if self.trialBeginFcn is not None:
                self.trialBeginFcn(self.rlParent.current_trial - 1, self.rlParent)

        def on_episode_end(self, epoch, logs):
            # update trial count
            self.rlParent.current_trial += 1
            self.rlParent.session_trial += 1
            # stop training after the maximum number of trials was reached
            if self.rlParent.session_trial >= self.rlParent.max_trials:
                self.rlParent.step = self.rlParent.max_steps + 1
            if self.trialEndFcn is not None:
                self.trialEndFcn(self.rlParent.current_trial - 1, self.rlParent, logs)

    def __init__(self, modules, externalMem, replay_type, epsilon=0.1, gamma=0.95, online_learning=False,
                 with_replay=True, trialBeginFcn=None, trialEndFcn=None):
        # store the Open AI Gym interface
        self.interfaceOAI = modules['rl_interface']
        # env dict
        self.obsDict = modules['world'].env.copy()
        # extract the input shape
        self.observation_space = np.array(list(self.obsDict.values())[0]).shape

        if len(self.observation_space) == 1:   # the inputs are vectors
            self.vectorInput = True
        else:      # the inputs are images
            self.vectorInput = False

        self.numberOfStates = modules['world'].numOfStates()
        # if the agent is also trained with the online experience
        self.online_learning = online_learning
        # if there is replay
        self.with_replay = with_replay

        self.batch_size = 32

        # the type of replay
        self.replay_type = replay_type
        # define the maximum number of steps
        self.max_steps = 10 ** 10
        # keeps track of current trial
        self.current_trial = 0  # trial count across all sessions (i.e. calls to the train/simulate method)
        self.session_trial = 0  # trial count in current seesion (i.e. current call to the train/simulate method)
        # define the maximum number of trials
        self.max_trials = 0
        ### Initializers for SFMA side
        # logging
        self.logs = {}
        self.logs['rewards'] = []
        self.logs['steps'] = []
        self.logs['behavior'] = []
        self.logs['modes'] = []
        self.logs['errors'] = []
        self.logs['replay_traces'] = {'start': [], 'end': []}
        self.logging_settings = {}
        for log in self.logs:
            self.logging_settings[log] = False
        # training
        self.online_replays_per_trial = 5  # number of replay batches which start from the terminal state
        self.offline_replays_per_trial = 5  # number of offline replay batches
        self.update_per_batch = 1 # number of updates for one single batch
        self.random_replay = False  # if true, random replay batches are sampled
        # retrieve number of actions
        self.numberOfActions = self.interfaceOAI.action_space.n
        # build model and target model
        self.build_model()
        self.target_model_update = 1e-2
        #define the available policies
        self.policy = EpsGreedyQPolicy(epsilon)
        # self.policy = BoltzmannQPolicy(tau=0.1)
        self.test_policy = self.policy

        self.memory = externalMem
        self.gamma = gamma
        # Meanwhile, we also initilize a sequential memory for random replay
        self.random_memory = SequentialMemory_adapted(limit=100000, window_length=1)
        # compile the model
        self.lr = 0.0001
        self.compile(Adam(lr=self.lr), metrics=['mse'])

        self.engagedCallbacks = self.callbacks(self, trialBeginFcn, trialEndFcn)

        self.recent_observation = None
        self.recent_action = None

    def build_model(self):
        if not self.vectorInput:
            # # We use the same CNN that was described by Mnih et al. (2015)
            sensory_model = Sequential()
            sensory_model.add(Convolution2D(16, kernel_size=(8, 8), strides=4, activation='relu',
                                            input_shape=self.observation_space))
            sensory_model.add(Convolution2D(32, kernel_size=(4, 4), strides=2, activation='relu'))
            sensory_model.add(Flatten())  # dimension: 3136
            feature_input = sensory_model.output
            x = Dense(256, activation='relu')(feature_input)
            x = Dense(self.numberOfActions, activation='linear')(x)
            self.model = Model(inputs=sensory_model.input, outputs=x)
        else:
            self.model = Sequential(
                [
                    Dense(256, activation='relu', input_shape=self.observation_space),
                    Dense(128, activation='relu'),
                    Dense(self.numberOfActions, activation='linear')
                ]
            )
        # build the target model
        self.target_model = clone_model(self.model)

    def compile(self, optimizer, metrics):
        self.model.compile(optimizer=optimizer, loss='huber_loss', metrics=metrics)
        self.target_model.compile(optimizer=optimizer, loss='mse', metrics=metrics)

    def train(self, numberOfTrials=100, maxNumberOfSteps=50):
        self.training = True
        self.max_trials = numberOfTrials
        self.session_trial = 0
        # record the total elapsed steps
        elapsed_steps = 0
        for trial in range(numberOfTrials):
            # reset environment
            state = self.interfaceOAI.reset()
            # log cumulative reward
            trial_log = {'rewards': 0, 'steps': 0, 'modes': None, 'errors': 0, 'behavior': []}
            for step in range(maxNumberOfSteps):
                # determine next action
                action, _ = self.forward(state)
                # perform action
                next_state, reward, terminal, _ = self.interfaceOAI.step(action)
                stopEpisode = terminal
                if step==maxNumberOfSteps-1:
                    stopEpisode = True
                self.backward(next_state, reward, terminal, stopEpisode)
                state = next_state
                # log behavior and reward
                trial_log['behavior'] += [[state['observationIdx'], action]]
                trial_log['rewards'] += reward
                if terminal:
                    break
            # log step and difference to optimal Q-function
            trial_log['steps'] = step
            elapsed_steps += step
            print('%s/%s Episode step: %s   Elapsed step: %s' % (trial, numberOfTrials, step, elapsed_steps))
            # store trial logs
            for log in trial_log:
                if self.logging_settings[log]:
                    self.logs[log] += [trial_log[log]]
            # callback
            self.engagedCallbacks.on_episode_end(trial, {'episode_reward': trial_log['rewards'],
                                                         'nb_episode_steps': trial_log['steps'],
                                                         'nb_steps': elapsed_steps})

    def backward(self, next_state, reward, terminal, stopEpisode):
        # make experience with state idx, not the real state
        experience = {'state': self.recent_state['observationIdx'], 'action': self.recent_action, 'reward': reward,
                      'next_state': next_state['observationIdx'], 'terminal': terminal}

        # train the agent online if needed
        if self.online_learning:
            self.update_network([experience])

        # store experience in SFMA memory
        self.memory.store(experience)
        # at the same time, store the experience in the sequential random memory, too
        self.random_memory.append(self.recent_state['observationIdx'], self.recent_action, reward, stopEpisode, training=True)
        # this is only for random memory storage. From keras-rl:
        if stopEpisode:
            # We are in a terminal state but the agent hasn't yet seen it. We therefore
            # perform one more forward-backward call and simply ignore the action before
            # resetting the environment. We need to pass in `terminal=False` here since
            # the *next* state, that is the state of the newly reset environment, is
            # always non-terminal by convention.
            self.forward(next_state)
            self.random_memory.append(self.recent_state['observationIdx'], self.recent_action, 0.0, False, training=True)

        if stopEpisode and self.with_replay:
            # offline replay multiple times
            if self.replay_type == 'SR_SU':
                for _ in range(self.online_replays_per_trial):
                    self.update_sequential(self.batch_size, next_state['observationIdx'])
                for _ in range(self.offline_replays_per_trial):
                    self.update_sequential(self.batch_size)

            elif self.replay_type == 'SR_AU':
                for _ in range(self.online_replays_per_trial):
                    self.update_average(self.batch_size, next_state['observationIdx'])
                for _ in range(self.offline_replays_per_trial):
                    self.update_average(self.batch_size)

            elif self.replay_type == 'RR_SU':
                for _ in range(self.online_replays_per_trial+self.offline_replays_per_trial):
                    replay_batch = self.random_memory.sample(self.batch_size)
                    if self.logging_settings['replay_traces']:
                        self.logs['replay_traces']['end'] += [replay_batch]
                    for _ in range(self.update_per_batch):
                        for item in replay_batch:
                            self.update_network([item])

            elif self.replay_type == 'RR_AU':
                for _ in range(self.online_replays_per_trial+self.offline_replays_per_trial):
                    replay_batch = self.random_memory.sample(self.batch_size)
                    if self.logging_settings['replay_traces']:
                        self.logs['replay_traces']['end'] += [replay_batch]
                    for _ in range(self.update_per_batch):
                        self.update_network(replay_batch)

    def replay(self, replayBatchSize=200, state=None):
        # sample batch of experiencess
        if self.memory.beta == 0:
            self.random_replay = True
        if self.random_replay:
            mask = np.array(self.memory.C != 0)
            replayBatch = self.memory.retrieve_random_batch(replayBatchSize, mask, False)
        else:
            replayBatch = self.memory.replay(replayBatchSize, state)
        return replayBatch

    def update_average(self, replayBatchSize, currentState=None):
        replay_batch = self.replay(replayBatchSize, currentState)
        if self.logging_settings['replay_traces']:
            self.logs['replay_traces']['end'] += [replay_batch]
        for _ in range(self.update_per_batch):
            self.update_network(replay_batch)

    def update_local(self, replayBatchSize, currentState=None):
        '''
        Update Q network by averaging each replay batch
        '''
        for _ in range(self.replays_per_trial):
            replay_batch = self.replay(replayBatchSize, currentState)
            if self.logging_settings['replay_traces']:
                self.logs['replay_traces']['end'] += [replay_batch]
            self.update_network(replay_batch, local_target=True)

    def update_sequential(self, replayBatchSize, currentState=None):
        replay_batch = self.replay(replayBatchSize, currentState)
        if self.logging_settings['replay_traces']:
            self.logs['replay_traces']['end'] += [replay_batch]
        for _ in range(self.update_per_batch):
            for item in replay_batch:
                self.update_network([item])

    def update_sequential_average(self, replayBatchSize, currentState=None):
        replay_batches = []
        minimum_batch_size = replayBatchSize
        for _ in range(self.replays_per_trial):
            replay_batch = self.replay(replayBatchSize, currentState)
            if self.logging_settings['replay_traces']:
                self.logs['replay_traces']['end'] += [replay_batch]
            replay_batches.append(replay_batch)
            minimum_batch_size = min(minimum_batch_size, len(replay_batch))
        for step in range(minimum_batch_size):
            experiences = []
            for batch in replay_batches:
                experiences.append(batch[step])
            self.update_network(experiences)

    def update_network(self, experiencebatch, local_target=False):
        # prepare placeholders for updating
        replay_size = len(experiencebatch)
        state0_batch, reward_batch, action_batch, terminal1_batch, state1_batch = [], [], [], [], []
        state0Idx_batch = []
        for e in experiencebatch:
            state0Idx_batch.append(e['state'])
            state0_batch.append(self.Idx2Observation(e['state']))
            state1_batch.append(self.Idx2Observation(e['next_state']))
            reward_batch.append(e['reward'])
            action_batch.append(e['action'])
            terminal1_batch.append(0 if e['terminal'] else 1)
        # Prepare parameters.
        # print(state0Idx_batch)
        state0_batch, state1_batch = np.array(state0_batch), np.array(state1_batch)
        terminal1_batch, reward_batch = np.array(terminal1_batch), np.array(reward_batch)

        # we first initialize the target values the same as the predictions over the current states, and then compute the real target values,
        # assign it to the Q value of the selected action. In this way, all other Q values got cancelled out.
        q_targets = self.model.predict_on_batch(state0_batch)
        if not local_target:
            # get q values from the target network
            target_q_values = self.target_model.predict_on_batch(state1_batch)
            q_batch = np.max(target_q_values, axis=1).flatten()
            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = self.gamma * q_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch
            Rs = reward_batch + discounted_reward_batch
        else:
            # we compute Q targets locally to propagate the reward infomation
            observations = self.get_all_observations()
            # observations = np.eye(self.numberOfStates)
            Q_local = self.target_model.predict_on_batch(observations)
            Rs = []
            for e in experiencebatch:
                # compute target
                target = e['reward']
                target += self.gamma * e['terminal'] * np.amax(Q_local[e['next_state']])
                # update local Q-function
                Q_local[e['state']][e['action']] = target
                Rs.append(target)
        Rs = np.asarray(Rs)
        # prepare varibles for updating
        for idx, (q_target, R, action) in enumerate(zip(q_targets, Rs, action_batch)):
            q_target[action] = R  # update action with estimated accumulated reward
        # update the model
        metrics = self.update_model(state0_batch, q_targets)

        return metrics

    def update_model(self, observations, targets, number_of_updates=1):
        '''
        This function updates the model on a batch of experiences.

        | **Args**
        | observations:                 The observations.
        | targets:                      The targets.
        | number_of_updates:            The number of backpropagation updates that should be performed on this batch.
        '''
        # update online model
        for update in range(number_of_updates):
            metrics = self.model.train_on_batch(observations, targets)
        # update target model by blending it with the online model
        weights_target = np.array(self.target_model.get_weights(), dtype=object)
        weights_online = np.array(self.model.get_weights(), dtype=object)
        weights_target += self.target_model_update * (weights_online - weights_target)
        self.target_model.set_weights(weights_target)
        return metrics

    def Idx2Observation(self, ObservationIdx):
        '''
        This function convert a list of state indice to real states by querying the world module
        '''
        obsIdxList = list(self.obsDict.keys())
        obsKey = obsIdxList[ObservationIdx]
        return self.obsDict[obsKey]

    def get_all_observations(self):
        observations = list(self.obsDict.values())[:-3]
        return np.asarray(observations)

    def forward(self, observation):
        # Select an action based on the state oberservation (ususally an image)
        state_idx = observation['observationIdx']
        state = self.Idx2Observation(state_idx)
        q_values = self.model.predict_on_batch(np.array([state])).flatten()
        if self.training:
            action = self.policy.select_action(q_values=q_values)
        else:
            action = self.test_policy.select_action(q_values=q_values)

        self.recent_state = observation
        self.recent_action = action

        return action, q_values

    def compute_q_value(self, batch):
        '''
        This function retrieves Q-values for a batch of states.

        | **Args**
        | batch:                        The batch of states.
        '''
        return self.model.predict_on_batch(batch)

    def book_keeping(self, state, action):
        '''
        Record the most recent obs and action
        '''
        self.recent_state = state
        self.recent_action = action
