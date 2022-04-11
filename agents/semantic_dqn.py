import warnings
import numpy as np
import tensorflow.compat.v1.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Layer, Dense
from sys import getsizeof
from rl_adapted.core import Agent
from rl_adapted.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl_adapted.util import *
from rl_adapted.memory import SequentialMemory, PrioritizedMemory, NoMemory, NStepMemory
from rl_adapted.agents.dqn import AbstractDQNAgent

def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))

class DQNAgent(AbstractDQNAgent):
    """
    """
    def __init__(self, model, policy=None, test_policy=None, enable_double_dqn=False, enable_dueling_network=False, with_replay=True,
                 dueling_type='avg', *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)

        # Validate (important) input.
        if list(model.output.shape) != list((None, self.nb_actions)):
            raise ValueError('Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action, in this case {}.'.format(model.output, self.nb_actions))

        # Parameters.
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type
        if self.enable_dueling_network:
            # get the second last layer of the model, abandon the last layer
            layer = model.layers[-2]
            nb_action = model.output.shape[-1]

            y = Dense(nb_action + 1, activation='linear')(layer.output)

            if self.dueling_type == 'avg':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(nb_action,))(y)
            elif self.dueling_type == 'max':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], axis=1, keepdims=True), output_shape=(nb_action,))(y)
            elif self.dueling_type == 'naive':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_action,))(y)
            else:
                assert False, "dueling_type must be one of {'avg','max','naive'}"

            model = Model(inputs=model.input, outputs=outputlayer)

        # Related objects.
        self.model = model
        if policy is None:
            policy = EpsGreedyQPolicy()
        if test_policy is None:
            test_policy = GreedyQPolicy()
        self.policy = policy
        self.test_policy = test_policy

        # State.
        self.reset_states()

        # the model always has a online buffer for online learning
        self.ol_memory = NoMemory(window_length=1)

        # flag for memory types with different updating rules
        if isinstance(self.memory, PrioritizedMemory):
            self.memory_type = 'prioritized'
        elif isinstance(self.memory, SequentialMemory):
            self.memory_type = 'sequential'
        elif isinstance(self.memory, NStepMemory):
            self.memory_type = 'nstep'

        # whether memory replay is activated
        self.with_replay = with_replay


    def get_config(self):
        config = super(DQNAgent, self).get_config()
        config['enable_double_dqn'] = self.enable_double_dqn
        config['dueling_type'] = self.dueling_type
        config['enable_dueling_network'] = self.enable_dueling_network
        config['model'] = get_object_config(self.model)
        config['policy'] = get_object_config(self.policy)
        config['test_policy'] = get_object_config(self.test_policy)
        if self.compiled:
            config['target_model'] = get_object_config(self.target_model)
        return config

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')

        # Compile model.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def clipped_masked_error(args):
            y_true, y_pred, mask = args
            loss = huber_loss(y_true, y_pred, self.delta_clip)
            loss *= mask  # apply element-wise mask
            return K.sum(loss, axis=-1)

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        y_pred = self.model.output
        y_true = Input(name='y_true', shape=(self.nb_actions,))
        mask = Input(name='mask', shape=(self.nb_actions,))
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_true, y_pred, mask])
        ins = [self.model.input] if type(self.model.input) is not list else self.model.input
        trainable_model = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model

        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.model.reset_states()
            self.target_model.reset_states()

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        q_values = self.compute_q_values(state)

        if self.training:
            action = self.policy.select_action(q_values=q_values)
        else:
            action = self.test_policy.select_action(q_values=q_values)
        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action
        return action


    def backward(self, reward, terminal):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)
            self.ol_memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]


        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # perform online learning at each step
        ol_experiences = self.ol_memory.sample()
        metrics = self.update_Qnet(ol_experiences, batch_size=1)

        # perform experience replay for once if replay learning is activated
        if self.with_replay:
            if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
                metrics = self.replay()

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    def replay(self):
        # Train the network on a single stochastic batch.
        if self.memory_type=='prioritized':
            # Calculations for current beta value based on a linear schedule.
            current_beta = self.memory.calculate_beta(self.step)
            # Sample from the memory.
            experiences = self.memory.sample(self.batch_size, current_beta, self.n_step, self.gamma)
        elif self.memory_type=='sequential':
            #SequentialMemory
            experiences = self.memory.sample(self.batch_size)
        elif self.memory_type=='nstep':
            experiences = self.memory.sample()
            self.batch_size = len(experiences) # for code consistency, no real meaning

        if len(experiences) == 0:
            return [np.nan for _ in self.metrics_names]

        metrics = self.update_Qnet(experiences, self.batch_size)

        return metrics


    def update_Qnet(self, experiences, batch_size):
        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state0_batch = []
        state1_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        importance_weights = []
        # We will be updating the idxs of the priority trees with new priorities
        pr_idxs = []

        if self.memory_type=='prioritized':
            for e in experiences[:-2]: # Prioritized Replay returns Experience tuple + weights and idxs.
                state0_batch.extend(e.state0)
                state1_batch.extend(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)
            importance_weights = experiences[-2]
            pr_idxs = experiences[-1]
        else: #SequentialMemory
            for e in experiences:
                state0_batch.extend(e.state0)
                state1_batch.extend(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

        if state1_batch[0] is None or state0_batch[0] is None: # at the first step of online learning in each episode, the next state has not been observed so none,
            return [np.nan for _ in self.metrics_names]               # then we skip the backward step


        state0_batch = self.process_state_batch(state0_batch)
        state1_batch = self.process_state_batch(state1_batch)

        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)
        assert reward_batch.shape == (batch_size,)
        assert terminal1_batch.shape == reward_batch.shape
        assert len(action_batch) == len(reward_batch)

        # Compute Q values for mini-batch update.
        if self.enable_double_dqn:
            # According to the paper "Deep Reinforcement Learning with Double Q-learning"
            # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
            # while the target network is used to estimate the Q value.
            q_values = self.model.predict_on_batch(state1_batch)
            assert q_values.shape == (batch_size, self.nb_actions)
            actions = np.argmax(q_values, axis=1)
            assert actions.shape == (batch_size,)
            # Now, estimate Q values using the target network but select the values with the
            # highest Q value wrt to the online model (as computed above).
            target_q_values = self.target_model.predict_on_batch(state1_batch)
            assert target_q_values.shape == (batch_size, self.nb_actions)
            q_batch = target_q_values[range(batch_size), actions]
        else:
            # Compute the q_values given state1, and extract the maximum for each sample in the batch.
            # We perform this prediction on the target_model instead of the model for reasons
            # outlined in Mnih (2015). In short: it makes the algorithm more stable.
            target_q_values = self.target_model.predict_on_batch(state1_batch)
            assert target_q_values.shape == (batch_size, self.nb_actions)
            q_batch = np.max(target_q_values, axis=1).flatten()
        assert q_batch.shape == (batch_size,)

        # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
        # but only for the affected output units (as given by action_batch).
        discounted_reward_batch = self.gamma * q_batch
        # Set discounted reward to zero for all states that were terminal.
        discounted_reward_batch *= terminal1_batch
        assert discounted_reward_batch.shape == reward_batch.shape
        # if a n-step online memory is used, we calculate the discounted accumulative rewards
        # of the next k, 1<=k<=n steps (n is normally small, e.g. 10) for each state
        if self.memory_type=='nstep':
            for i in range(len(reward_batch)-1):
                discounted_reward_batch[i] *= self.gamma**(len(reward_batch)-i-2)
                for j in range(i+1, len(reward_batch)):
                    reward_batch[i]+=self.gamma**(j-i)*reward_batch[j]
            # during update, we only update 1 sample
            state0_batch = np.array([state0_batch[0]])
            state1_batch = np.array([state1_batch[0]])
            action_batch = np.array([action_batch[0]])
            if terminal1_batch[0]:  # here it actually the final state is not terminal
                reward_batch = [reward_batch[0]-reward_batch[-1]]
            else:
                reward_batch = [reward_batch[0]]
            reward_batch = np.array(reward_batch)
            discounted_reward_batch = np.array([discounted_reward_batch[0]])

        targets = np.zeros((batch_size, self.nb_actions))
        dummy_targets = np.zeros((batch_size,))
        masks = np.zeros((batch_size, self.nb_actions))

        Rs = reward_batch + discounted_reward_batch
        for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
            target[action] = R  # update action with estimated accumulated reward
            dummy_targets[idx] = R
            mask[action] = 1.  # enable loss for this specific action
        targets = np.array(targets).astype('float32')
        masks = np.array(masks).astype('float32')

        if self.memory_type!='prioritized':
            importance_weights = [1. for _ in range(batch_size)]
        #Make importance_weights the same shape as the other tensors that are passed into the trainable model
        assert len(importance_weights) == batch_size
        importance_weights = np.array(importance_weights)
        importance_weights = np.vstack([importance_weights]*self.nb_actions)
        importance_weights = np.reshape(importance_weights, (batch_size, self.nb_actions))
        # Finally, perform a single update on the entire batch. We use a dummy target since
        # the actual loss is computed in a Lambda layer that needs more complex input. However,
        # it is still useful to know the actual target to compute metrics properly.
        ins = [state0_batch] if type(self.model.input) is not list else state0_batch
        metrics = self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets])

        if self.memory_type=='prioritized':
            assert len(pr_idxs) == batch_size
            #Calculate new priorities.
            y_true = targets
            y_pred = self.model.predict_on_batch(ins)
            #Proportional method. Priorities are the abs TD error with a small positive constant to keep them from being 0.
            new_priorities = (abs(np.sum(y_true - y_pred, axis=-1))) + 1e-5
            assert len(new_priorities) == batch_size
            #update priorities
            self.memory.update_priorities(pr_idxs, new_priorities)

        metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]  # throw away individual losses
        metrics += self.policy.metrics
        if self.processor is not None:
            metrics += self.processor.metrics

        return metrics

    @property
    def layers(self):
        return self.model.layers[:]

    @property
    def metrics_names(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_model.output_names) == 2
        dummy_output_name = self.trainable_model.output_names[1]
        model_metrics = [name for idx, name in enumerate(self.trainable_model.metrics_names) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    @property
    def policy(self):
        return self.__policy

    @policy.setter
    def policy(self, policy):
        self.__policy = policy
        self.__policy._set_agent(self)

    @property
    def test_policy(self):
        return self.__test_policy

    @test_policy.setter
    def test_policy(self, policy):
        self.__test_policy = policy
        self.__test_policy._set_agent(self)
