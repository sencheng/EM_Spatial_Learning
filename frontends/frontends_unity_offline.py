import sys
import os
import numpy as np
import pickle
import time
from shapely.geometry import Polygon
from mlagents_envs.base_env import ActionTuple

class FrontendUnityInterface():
    '''
    The Unity interface class. This is tailored for offline simulation
    Parameters:
        scenarioName: the name of the scenario that should be processed
        step_size:    the distance between two nearby node.
    '''
    def __init__(self, scenarioName=None, step_size=1.0):
        # load the saved images for the environment
        current_path = os.path.dirname(os.path.abspath(__file__))
        env_path = current_path + '/../environments/offline_unity/%s_ss%s_Infos.pickle' % (scenarioName, step_size)
        try:
            with open(env_path, 'rb') as handle:
                data = pickle.load(handle)
        except FileNotFoundError:
            raise Exception('Offline environment for %s, step_size=%s does not exist, please use script environments/generate_off_unity.py to generate it.' % (scenarioName, step_size))
        # Here env is a dict containing the topology idx as keys and corresponding images as values
        self.env = data
        self.world_limits = data['world_limits']
        self.walls_limits = data['walls_limits']
        self.perimeter_nodes = data['perimeter_nodes']

        # a dict that stores environmental information in each time step
        self.envData=dict()
        self.envData['poseData']=None
        self.envData['imageData']=None

        # flag to tell other modules that it is now offline mode
        self.offline = True

        self.scenarioName = scenarioName

    def setTopology(self,topologyModule):
        '''
        This function supplies the interface with a valid topology module
        topologyModule: the topologyModule to be supplied
        '''
        self.topologyModule=topologyModule


    def step_simulation_without_physics(self, newNode, newYaw):
        '''
        This function propels the simulation. It uses teleportation to guide the agent/robot directly by means of global x,y,yaw values.
        newX:       the global x position to teleport to
        newZ:       the global z position to teleport to
        yaw:     the global yaw value to teleport to
        '''
        # get the camera observation, time of the simulation and pose of the robot
        imageData = self.env[str([newNode, newYaw])]
        poseData = np.zeros(3) # Here 3 dimension is only for being compatible with other modules
        poseData[2] = newYaw

        # update environmental information
        self.envData['imageData']=imageData
        self.envData['poseData']=poseData

        # return acquired data
        return [imageData, poseData]

    def actuateRobot(self,actuatorCommand):
        '''
        This function actually actuates the agent/robot in the virtual environment.
        actuatorCommand:  the command that is used in the actuation process
        '''
        # call the teleportation routine
        [imageData, poseData]=self.step_simulation_without_physics(actuatorCommand[0],actuatorCommand[1])

        # return the data acquired from the robot/agent/environment
        return imageData, poseData

    def getLimits(self):
        '''
         This function returns the limits of the environmental perimeter.
        '''
        # rearrange the elements for topology module to use
        world_limits = [[self.world_limits[0], self.world_limits[2]], [self.world_limits[1], self.world_limits[3]]]
        return np.asarray(world_limits)

    def getWallGraph(self):
        '''
         This function returns the environmental perimeter by means of wall vertices/segments.
        '''
        return self.walls_limits, self.perimeter_nodes

    def stopUnity(self):
        del self.env
