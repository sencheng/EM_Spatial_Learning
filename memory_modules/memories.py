# basic imports
import numpy as np
from collections import deque, namedtuple
# keras imports
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
# keras-rl imports
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import Memory, RingBuffer, sample_batch_indexes
from rl.core import Processor

Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')

class simpleRing(object):
    def __init__(self, maxlen):
        self.maxlen=maxlen
        self.start=0
        self.length=0
        self.data=[None for _ in range(maxlen)]

    def append(self, v):
        self.data[self.start]=v
        self.start=(self.start+1) % self.maxlen
        self.length+=1
        self.length=min(self.length, self.maxlen)
        return self.start

    def __getitem__(self, idx):
        return self.data[idx % self.maxlen]

    def renew(self):
        self.data=[None for _ in range(self.maxlen)]
        self.start=0
        self.length=0

class NoMemory(Memory):
    '''
    An online memory buffer that only stores the most recent experience tuple
    '''
    def __init__(self, **kwargs):
        super(NoMemory, self).__init__(**kwargs)

        self.actions = simpleRing(2)
        self.rewards = simpleRing(2)
        self.terminals = simpleRing(2)
        self.observations = simpleRing(2)
        self.steps = simpleRing(2)
        self.append_times=0

    def append(self, observation, action, reward, terminal, training=True):
        super(NoMemory, self).append(observation, action, reward, terminal, training=training)

        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.append_idx=self.terminals.append(terminal) # get the idx of the latest stored experience

    def sample(self, batch_size=0): # batch_size is not used here, only for compatible with the keras-rl package
        # here we sample only the transition at the current step
        state0=self.observations[(self.append_idx) % 2] # retrive the state before the newest appended one, -2 because indice starts with 0
        state1=self.observations[(self.append_idx-1) % 2]  # the state appended the most recent
        action=self.actions[(self.append_idx) % 2]
        reward=self.rewards[(self.append_idx) % 2]
        terminal=self.terminals[(self.append_idx) % 2]

        experiences=[]
        experiences.append(Experience(state0=state0, action=action, reward=reward,
                                      state1=state1, terminal1=terminal))

        if terminal:  # during online dqn, we renew the buffer at the end of each episode
            self.actions.renew()
            self.rewards.renew()
            self.terminals.renew()
            self.observations.renew()

        return experiences

class SequentialMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(SequentialMemory, self).__init__(**kwargs)
        
        self.limit = limit

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        """Return a randomized batch of experiences

        # Argument
            batch_size (int): Size of the all batch
            batch_idxs (int): Indexes to extract
        # Returns
            A list of experiences randomly selected
        """
        # It is not possible to tell whether the first state in the memory is terminal, because it
        # would require access to the "terminal" flag associated to the previous state. As a result
        # we will never return this first state (only using `self.terminals[0]` to know whether the
        # second state is terminal).
        # In addition we need enough entries to fill the desired window length.
        assert self.nb_entries >= self.window_length + 2, 'not enough entries in the memory'

        if batch_idxs is None:
            # Draw random indexes such that we have enough entries before each index to fill the
            # desired window length.
            batch_idxs = sample_batch_indexes(
                self.window_length, self.nb_entries - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= self.window_length + 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2]
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                idx = sample_batch_indexes(self.window_length + 1, self.nb_entries, size=1)[0]
                terminal0 = self.terminals[idx - 2]
            assert self.window_length + 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                assert current_idx >= 1
                current_terminal = self.terminals[current_idx - 1]
                if current_terminal and not self.ignore_episode_boundaries:
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [copy.deepcopy(x) for x in state0[1:]]
            state1.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0[0], action=action, reward=reward,
                                          state1=state1[0], terminal1=terminal1))
        assert len(experiences) == batch_size
        return experiences

    def append(self, observation, action, reward, terminal, training=True):
        """Append an observation to the memory

        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """ 
        super(SequentialMemory, self).append(observation, action, reward, terminal, training=training)
        
        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    @property
    def nb_entries(self):
        """Return number of observations

        # Returns
            Number of observations
        """
        return len(self.observations)

    def get_config(self):
        """Return configurations of SequentialMemory

        # Returns
            Dict of config
        """
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config



class SequentialMemory_adapted(Memory):
    def __init__(self, limit, **kwargs):
        super(SequentialMemory_adapted, self).__init__(**kwargs)

        self.limit = limit

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        """Return a randomized batch of experiences

        # Argument
            batch_size (int): Size of the all batch
            batch_idxs (int): Indexes to extract
        # Returns
            A list of experiences randomly selected
        """
        # It is not possible to tell whether the first state in the memory is terminal, because it
        # would require access to the "terminal" flag associated to the previous state. As a result
        # we will never return this first state (only using `self.terminals[0]` to know whether the
        # second state is terminal).
        # In addition we need enough entries to fill the desired window length.
        assert self.nb_entries >= self.window_length + 2, 'not enough entries in the memory'

        if batch_idxs is None:
            # Draw random indexes such that we have enough entries before each index to fill the
            # desired window length.
            batch_idxs = sample_batch_indexes(
                self.window_length, self.nb_entries - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= self.window_length + 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2]
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                idx = sample_batch_indexes(self.window_length + 1, self.nb_entries, size=1)[0]
                terminal0 = self.terminals[idx - 2]
            assert self.window_length + 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                assert current_idx >= 1
                current_terminal = self.terminals[current_idx - 1]
                if current_terminal and not self.ignore_episode_boundaries:
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [copy.deepcopy(x) for x in state0[1:]]
            state1.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append({'state':state0[0], 'action':action, 'reward':reward,
                                'next_state': state1[0], 'terminal':terminal1})
        assert len(experiences) == batch_size
        return experiences

    def append(self, observation, action, reward, terminal, training=True):
        """Append an observation to the memory

        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """
        super(SequentialMemory_adapted, self).append(observation, action, reward, terminal, training=training)

        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    @property
    def nb_entries(self):
        """Return number of observations

        # Returns
            Number of observations
        """
        return len(self.observations)

    def get_config(self):
        """Return configurations of SequentialMemory

        # Returns
            Dict of config
        """
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config



class SparseSequentialMemory(SequentialMemory):
    '''
    This memory module can keep sampling the same batch of experiences for a chosen sample interval
    '''
    def __init__(self, sample_interval, **kwargs):
        super(SparseSequentialMemory, self).__init__(**kwargs)
        self.sample_interval = sample_interval
        self.experiences = None
        self.sampled_step = 0

    def sample(self, batch_size, batch_idxs=None):
        if self.experiences is None:
            self.experiences = super().sample(batch_size, batch_idxs)
        else:
            if self.sampled_step == 0:
                self.experiences = super().sample(batch_size, batch_idxs)

        self.sampled_step += 1
        if self.sampled_step == self.sample_interval:
            self.sampled_step = 0

        return self.experiences












