import os
import time
import numpy as np
import gym
import subprocess
import signal
from gym import spaces
from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel
from rl_adapted.core import Processor

class OAIGymInterface(gym.Env):
    '''
    ### This is the Open AI gym interface class. The interface wraps the control path and ensures communication
    ### between the agent and the environment. The class descends from gym.Env, and is designed to be minimalistic (currently!).
    # The constructor.
    # modules:          the dict of all available modules in the system
    # withGUI:          if True, the module provides GUI control
    # rewardCallback:   this callback function is invoked in the step routine in order to get the appropriate reward w.r.t. the experimental design
    '''
    def __init__(self, modules, withGUI=True, rewardCallback=None):
        # store the modules
        self.modules = modules

        # store visual output variable
        self.withGUI = withGUI

        # memorize the reward callback function
        self.rewardCallback = rewardCallback

        self.world = self.modules['world']
        self.observations = self.modules['observation']

        # second: action space
        self.action_space = modules['spatial_representation'].get_action_space()

        # third: observation space
        self.observation_space = modules['observation'].getObservationSpace()

        # all OAI spaces have been initialized!

        # this observation variable is filled by the OBS modules
        self.observation = None

        # a variable that allows the OAI class to access the robotic agent class
        self.rlAgent = None

        # the current time step within an episode; can be understood as how much time
        # has passed in a trial
        self.episode_step = 0


    def updateObservation(self, observation):
        '''
        # This function (slot) updates the observation provided by the environment
        # observation:  the observation used to perform the update
        '''
        self.observation = observation

    def _step(self, action):
        '''
        # The step function that propels the simulation.
        # This function is called by the .fit function of the RL agent whenever a novel action has been computed.
        # The action is used to decide on the next topology node to run to, and step then triggers the control path (including 'Blender')
        # by making use of direct calls and signal/slot methods.
        #
        # action:   the action to be executed
        '''
        self.episode_step += 1
        callbackValue = self.modules['spatial_representation'].generate_behavior_from_action(action)
        callbackValue['rlAgent'] = self.rlAgent
        callbackValue['modules'] = self.modules
        callbackValue['episode_step'] = self.episode_step

        reward, stopEpisode = self.rewardCallback(callbackValue)

        return self.modules['observation'].observation, reward, stopEpisode, {}

    # This function restarts the RL agent's learning cycle by initiating a new episode.
    #
    def _reset(self):
        self.modules['spatial_representation'].generate_behavior_from_action('reset')
        self.episode_step = 0
        # return the observation
        return self.modules['observation'].observation

