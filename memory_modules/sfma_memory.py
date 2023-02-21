# basic imports
import numpy as np
import math
# framework imports
  
    
class SFMAMemory():
    '''
    Memory module to be used with the SMA agent.
    Experiences are stored as a static table.
    
    | **Args**
    | numberOfStates:               The number of states in the env.
    | numberOfActions:              The number of the agent's actions.
    | metric:						The metrics used for updating C and D matrix. Normally a DR metrics.  
    | decay_inhibition:             The factor by which inhibition is decayed.
    | decay_strength:               The factor by which the experience strengths are decayed.
    | learningRate:                 The learning rate with which reward experiences are updated.
    '''
    
    def __init__(self, numberOfStates, numberOfActions, metric, decay_inhibition=0.9, decay_strength=1., learningRate=1.0):
        # initialize variables
        self.numberOfStates = numberOfStates
        self.numberOfActions = numberOfActions
        self.decay_inhibition = decay_inhibition
        self.decay_strength = decay_strength
        self.decay_recency = 0.9
        self.learningRate = learningRate
        self.beta = 20
        self.rlAgent = None
        # experience strength modulation parameters
        self.reward_mod_local = False # increase during experience
        self.error_mod_local = False # increase during experience
        self.reward_mod = False # increase during experience
        self.error_mod = False # increase during experience
        self.policy_mod = False # added before replay
        self.state_mod = False # 
        # similarity metric
        self.metric = metric
        # prepare memory structures
        self.rewards = np.zeros((self.numberOfStates, self.numberOfActions))
        self.states = np.tile(np.arange(self.numberOfStates).reshape(self.numberOfStates, 1), self.numberOfActions).astype(int)
        self.terminals = np.zeros((self.numberOfStates, self.numberOfActions)).astype(int)
        # prepare replay-relevant structures
        self.C = np.zeros(self.numberOfStates * self.numberOfActions) # strength
        self.T = np.zeros(self.numberOfStates * self.numberOfActions) # recency
        self.I = np.zeros(self.numberOfStates) # inhibition
        # increase step size
        self.C_step = 1.
        self.I_step = 1.
        # priority rating threshold
        self.R_threshold = 0.
        # always reactive experience with highest priority rating
        self.deterministic = False
        # consider recency of experience
        self.recency = False
        # normalize variables
        self.C_normalize = True
        self.D_normalize = True
        self.R_normalize = True
        # replay mode
        self.mode = 'default'
        # modulates reward
        self.reward_modulation = 1.
        # weighting of forward/reverse mode when using blending modes
        self.blend = 0.1
        # weightings of forward abdreverse modes when using interpolation mode
        self.interpolation_fwd, self.interpolation_rev = 0.5, 0.5
        
    def store(self, experience):
        '''
        This function stores a given experience.
        
        | **Args**
        | experience:                   The experience to be stored.
        '''
        state, action = experience['state'], experience['action']
        # update experience
        self.rewards[state][action] += self.learningRate * (experience['reward'] - self.rewards[state][action])
        self.states[state][action] = experience['next_state']
        self.terminals[state][action] = experience['terminal']
        # update replay-relevent structures
        self.C *= self.decay_strength
        self.C[self.numberOfStates * action + state] += self.C_step
        self.T *= self.decay_recency
        self.T[self.numberOfStates * action + state] = 1.
        # local reward modulation (affects this experience only)
        if self.reward_mod_local:
            self.C[self.numberOfStates * action + state] += experience['reward'] * self.reward_modulation
        # reward modulation (affects close experiences)
        if self.reward_mod:
            modulation = np.tile(self.metric.D[experience['state']], self.numberOfActions)
            self.C += experience['reward'] * modulation * self.reward_modulation
        # local RPE modulation (affects this experience only)
        if self.error_mod_local:
            self.C[self.numberOfStates * action + state] += np.abs(experience['error'])
        # RPE modulation (affects close experiences)
        if self.error_mod:
            modulation = np.tile(self.metric.D[experience['next_state']], self.numberOfActions)
            self.C += np.abs(experience['error']) * modulation
        # additional strength increase of all experiences at current state
        if self.state_mod:
            self.C[[state + self.numberOfStates * a for a in range(self.numberOfActions)]] += 1.
    
    def replay(self, replayLength, current_state=None, current_action=None, offline=False):
        '''
        This function replays experiences.
        
        | **Args**
        | replayLength:                 The number of experiences that will be replayed.
        | current_state:                State at which replay should start.
        | current_action:               Action with which replay should start.
        '''
        action = current_action
        # if no action is specified pick one at random
        if current_action is None:
            action = np.random.randint(self.numberOfActions)
        # if a state is not defined, then choose an experience according to relative experience strengths
        if current_state is None:
            # we clip the strengths to catch negative values caused by rounding errors
            P = np.clip(self.C, a_min=0, a_max=None)/np.sum(np.clip(self.C, a_min=0, a_max=None))
            probs = self.softmax(P, -1, self.beta)
            exp = np.random.choice(np.arange(0, probs.shape[0]), p=probs)
            current_state = exp % self.numberOfStates
            action = int(exp/self.numberOfStates)
        next_state = self.states[current_state, action]
        # reset inhibition
        self.I *= 0
        # replay
        experiences = []
        for step in range(replayLength):
            # retrieve experience strengths
            C = np.copy(self.C)
            if self.C_normalize:
                C /= np.amax(C)
            # retrieve experience similarities
            D = np.tile(self.metric.D[current_state], self.numberOfActions)

            if self.D_normalize:
                D /= np.amax(D)
            if self.mode == 'forward':
                D = np.tile(self.metric.D[next_state], self.numberOfActions)
            elif self.mode == 'reverse':
                D = D[self.states.flatten(order='F')]
            elif self.mode == 'blend_forward':
                D += self.blend * np.tile(self.metric.D[next_state], self.numberOfActions)
            elif self.mode == 'blend_reverse':
                D += self.blend * D[self.states.flatten(order='F')]
            elif self.mode == 'interpolate':
                D = self.interpolation_fwd * np.tile(self.metric.D[next_state], self.numberOfActions) + self.interpolation_rev * D[self.states.flatten(order='F')]
            # retrieve inhibition
            I = np.tile(self.I, self.numberOfActions)
            # compute priority ratings
            R = C * D * (1 - I)

            if self.recency:
                R *= self.T
            # apply threshold to priority ratings
            R[R < self.R_threshold] = 0.
            # stop replay sequence if all priority ratings are all zero
            if np.sum(R) == 0.:
                break
            # determine state and action
            if self.R_normalize:
                R /= np.amax(R)

            exp = np.argmax(R)

            if not self.deterministic:
                # compute activation probabilities
                probs = self.softmax(R, -1.0, self.beta)
                probs = probs/np.sum(probs)
                exp = np.random.choice(np.arange(0,probs.shape[0]), p=probs)
            # determine experience tuple
            action = int(exp/self.numberOfStates)
            current_state = exp - (action * self.numberOfStates)
            next_state = self.states[current_state][action]
            # apply inhibition
            self.I *= self.decay_inhibition
            self.I[current_state] = min(self.I[current_state] + self.I_step, 1.)
            # "reactivate" experience
            experience = {'state': current_state, 'action': action, 'reward': self.rewards[current_state][action],
                          'next_state': next_state, 'terminal': self.terminals[current_state][action]}
            experiences += [experience]
            # stop replay at terminal states
            #if experience['terminal']:
            #    break
            
        return experiences
    

    def softmax(self, data, offset=0, beta=5):
        '''
        This function computes the softmax over the input.
        
        | **Args**
        | data:                         Input of the softmax function.
        | offset:                       Offset added after applying the softmax function.
        | beta:                         Beta value.
        '''
        exp = np.exp(data * beta) + offset

        if np.sum(exp) == 0:
            exp.fill(1)
        else:
            exp /= np.sum(exp)

        return exp
    
    def retrieve_random_batch(self, numberOfExperiences, mask, strength_based=False):
        '''
        This function retrieves a number of random experiences.
        
        | **Args**
        | numberOfExperiences:          The number of random experiences to be drawn.
        | if the random retrival should based on the strength of each experience
        '''
        if not strength_based:
            # draw random experiences
            probs = np.ones(self.numberOfStates * self.numberOfActions) * mask.astype(int)
            probs /= np.sum(probs)
        else:
            # retrieve experience strengths
            C = np.copy(self.C)
            probs = C / np.sum(C)

        idx = np.random.choice(np.arange(self.numberOfStates * self.numberOfActions), numberOfExperiences, p=probs)
        # determine indeces
        idx = np.unravel_index(idx, (self.numberOfStates, self.numberOfActions), order='F')
        # build experience batch
        experiences = []
        for exp in range(numberOfExperiences):
            state, action = idx[0][exp], idx[1][exp]
            experiences += [{'state': state, 'action': action, 'reward': self.rewards[state][action],
                             'next_state': self.states[state][action], 'terminal': self.terminals[state][action]}]
            
        return experiences

class SFMAGainMemory(SFMAMemory):

    def __init__(self, numberOfStates, numberOfActions, metric, decay_inhibition=0.9, decay_strength=1., learningRate=1.0):

        super().__init__(numberOfStates, numberOfActions, metric, decay_inhibition, decay_strength, learningRate)
        self.rl_agent = None
        # prepare replay-relevant structures
        self.G = np.zeros(self.numberOfStates * self.numberOfActions)  # gain estimates
        self.G_normalize = False
        self.min_gain = 10 ** -6

    def update_gain(self, experience: dict):
        '''
        This function updates the gain estimates.

        Parameters
        ----------
        experience :                        The experience tuple.

        Returns
        ----------
        None
        '''
        self.G[experience['state'] + experience['action'] * self.numberOfStates] = 0.
        self.G[self.states.flatten(order='F') == experience['state']] = experience['gain']

    def replay(self, replay_length: int, current_state=None, current_action=None) -> list:
        '''
        This function replays experiences.

        Parameters
        ----------
        replay_length :                     The number of experiences that will be replayed.
        current_state :                     The state at which replay should start.
        current_action :                    The action with which replay should start.

        Returns
        ----------
        experiences :                       The replay batch.
        '''
        action = current_action
        # if no action is specified pick one at random
        if current_action is None:
            action = np.random.randint(self.numberOfActions)
        # if a state is not defined, then choose an experience according to relative experience strengths
        if current_state is None:
            # we clip the strengths to catch negative values caused by rounding errors
            P = np.clip(self.C, a_min=0, a_max=None) / np.sum(np.clip(self.C, a_min=0, a_max=None))
            exp = np.random.choice(np.arange(0, P.shape[0]), p=P)
            current_state = exp % self.numberOfStates
            action = int(exp / self.numberOfStates)
        next_state = self.states[current_state, action]
        # reset inhibition
        self.I *= 0
        # replay
        experiences = []
        for step in range(replay_length):
            # retrieve experience strengths
            C = np.copy(self.C)
            if self.C_normalize:
                C /= np.amax(C)
            # retrieve gain estimates
            G = np.clip(np.copy(self.G), a_min=self.min_gain, a_max=None)
            # G = G/np.amax(G) + 1.
            if self.G_normalize:
                G /= np.amax(G)
            # retrieve experience similarities
            D = np.tile(self.metric.D[current_state], self.numberOfActions)
            if self.D_normalize:
                D /= np.amax(D)
            if self.mode == 'forward':
                D = np.tile(self.metric.D[next_state], self.numberOfActions)
            elif self.mode == 'reverse':
                D = D[self.states.flatten(order='F')]
            elif self.mode == 'blend_forward':
                D += self.blend * np.tile(self.metric.D[next_state], self.numberOfActions)
            elif self.mode == 'blend_reverse':
                D += self.blend * D[self.states.flatten(order='F')]
            elif self.mode == 'interpolate':
                D = self.interpolation_fwd * np.tile(self.metric.D[next_state],
                                                     self.numberOfActions) + self.interpolation_rev * D[
                        self.states.flatten(order='F')]
            # retrieve inhibition
            I = np.tile(self.I, self.numberOfActions)
            # compute priority ratings
            R = C * D * G * (1 - I)
            if self.recency:
                R *= self.T
            # apply threshold to priority ratings
            R[R < self.R_threshold] = 0.
            # stop replay sequence if all priority ratings are all zero
            if np.sum(R) == 0.:
                break
            # determine state and action
            if self.R_normalize:
                R /= np.amax(R)
            exp = np.argmax(R)
            if not self.deterministic:
                # compute activation probabilities
                probs = self.softmax(R, -1, self.beta)
                probs = probs / np.sum(probs)
                exp = np.random.choice(np.arange(0, probs.shape[0]), p=probs)
            # determine experience tuple
            action = int(exp / self.numberOfStates)
            current_state = exp - (action * self.numberOfStates)
            next_state = self.states[current_state][action]
            # apply inhibition
            self.I *= self.decay_inhibition
            self.I[current_state] = min(self.I[current_state] + self.I_step, 1.)
            # "reactivate" experience
            experience = {'state': current_state, 'action': action, 'reward': self.rewards[current_state][action],
                          'next_state': next_state, 'terminal': self.terminals[current_state][action]}
            experiences += [experience]
            #self.rl_agent.update_Q(experience)


        return experiences
