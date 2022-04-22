import sys
import socket
import os
import subprocess
import numpy as np
import time
from shapely.geometry import Polygon

from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel
from frontends.Env_sidechannel import Env_sidechannel
from mlagents_envs.base_env import ActionTuple

def get_env_path():
    """
    returns the unity env path
    TODO move this to some kind of utility class?
    """
    if 'UNITY_ENVIRONMENT_EXECUTABLE' in os.environ.keys():
        return os.environ['UNITY_ENVIRONMENT_EXECUTABLE']
    else:
        return None

### The Unity interface class. This class connects to the Unity environment and controls the flow of commands/data that goes to/comes from the Blender environment.
###
class FrontendUnityInterface():

    ### Constructor
    ### scenarioName: the name of the scenario that should be processed
    def __init__(self, scenarioName=None, side_channels=None, seed=42, timeout_wait=60, time_scale=3.0, worker_id=0, step_size=1.0):

        self.env_exec_path = None
        # setup side channels
        if side_channels is None:
            side_channels = []

        self.offline = False

        # Create the channel for sending world information of the Unity env
        self.env_channel = Env_sidechannel()
        side_channels.append(self.env_channel)

        # setup engine channel
        self.engine_configuration_channel = EngineConfigurationChannel()

        # add engine config channel
        side_channels.append(self.engine_configuration_channel)

        # step parameters
        self.env_configuration_channel = FloatPropertiesChannel()
        self.env_configuration_channel.set_property("max_step", 100000)


        # add env config channel
        side_channels.append(self.env_configuration_channel)

        # command line args
        args = []

        # when env is an executable mlagents can load a given scene.
        if scenarioName is not None:
            args = ["--mlagents-scene-name", scenarioName]

        self.scenarioName = scenarioName
        # select random worker id.
        # There is an issue in Linux where the worker id becomes available only after some time has passed since the
        # last usage. In order to mitigate the issue, a (hopefully) new worker id is automatically selected unless
        # specifically instructed not to.
        # Implementation note: two consecutive samples between 0 and 1200 have an immediate 1/1200 chance of
        # being the same. By using the modulo of unix time we arrive to that likelihood only after an hour, by when
        # the port has hopefully been released.
        # Additional notes: The ML-agents framework adds 5004 to the worker_id internally, so no need to worry about
        # port collision with the OS.
        if scenarioName is not None:
            # get the path to the unity environment executable
            self.env_exec_path=get_env_path()
            worker_id = round(time.time()) % 1200

        # try to start the communicator
        try:
            # connect python to executable environment
            env = UnityEnvironment(file_name=self.env_exec_path, worker_id=worker_id, seed=seed, base_port=5004,
                                   timeout_wait=timeout_wait, side_channels=side_channels, no_graphics=False,
                                   additional_args=args)
            # reset the environment
            env.reset()

            ## the world information has been sent from Unity to Python, env_info contains three strings corresponding
            ## to: (1) min and max XY values of the world (2) The wall information (3) The polygon that defines the perimeter
            ## of the maze
            env_info = self.env_channel.received_info
            # clear the received info, very important to do so; otherwise mess up the received info
            self.env_channel.clear_received_info()
            # [minx, minz, maxx, maxz]
            self.world_limits = [float(x) for x in env_info[0].split()]
            # extract the [minx, minz, maxx, maxz] for each wall
            self.walls_limits = []
            for item in env_info[1].split(","):
                wl = [float(x) for x in item.split()]
                self.walls_limits.append(wl)
            # extract the polygon nodes coordinates
            self.perimeter_nodes = []
            for item in env_info[2].split(","):
                p_node = [float(x) for x in item.split()]
                self.perimeter_nodes.append(p_node)
            self.perimeter_nodes.append(self.perimeter_nodes[0]) #repeat the first point to create a 'closed loop'

            self.world_limits = np.asarray(self.world_limits)
            self.walls_limits = np.asarray(self.walls_limits)
            self.perimeter_nodes = np.asarray(self.perimeter_nodes)

            # receive environment name from environment
            behavior_name = list(env.behavior_specs)[0]
            print(f"Name of the behavior : {behavior_name}")

            # save environment variables
            self.env = env
            self.behavior_name = behavior_name

        except UnityWorkerInUseException as e:
            print("the desired port is still in use. please retry after a few seconds.")

        # a dict that stores environmental information in each time step
        self.envData=dict()
        self.envData['poseData']=None
        self.envData['imageData']=None

        # initial robot pose of [X, Z, Yaw]; it is Z coordinate here because the agent is moving on the XZ plane
        self.robotPose=np.array([0.0,0.0,0.0])

    ### This function supplies the interface with a valid topology module
    ###
    ### topologyModule: the topologyModule to be supplied
    def setTopology(self,topologyModule):
        self.topologyModule=topologyModule

    # This function propels the simulation. It uses physics to guide the agent/robot with linear and angular velocities.
    #
    # velocityLinear:       the requested linear (translational) velocity of the agent/robot
    # omega:                the requested rotational (angular) velocity of the agent/robot
    #
    # Note: linear and angular velocities can be set simultaneously to force the robot to go in curved paths
    def stepSimulation(self,velocityLinear,omega):
        pass


    # This function propels the simulation. It uses teleportation to guide the agent/robot directly by means of global x,y,yaw values.
    #
    # x:       the global x position to teleport to
    # y:       the global y position to teleport to
    # yaw:     the global yaw value to teleport to
    #
    def step_simulation_without_physics(self,newX,newZ,newYaw):
        # send teleported positions to Unity and retrieve the observation
        # step the env with the provided action.
        # the orientation of the coordinate sys in Unity and here is different such that the sum of two is phase 90 degree
        action = ActionTuple(np.asarray([[newX, newZ, 90 - newYaw]]))
        self.env.set_actions(self.behavior_name, action)
        # forward the simulation by a tick (and execute action)
        self.env.step()

        # get the camera observation, time of the simulation and pose of the robot
        imageData, poseData = self.__get_step_results()

        # update robot's/agent's pose
        self.robotPose=np.array(poseData)
        # update environmental information
        self.envData['imageData']=imageData
        self.envData['poseData']=poseData

        # return acquired data
        return [imageData, poseData]

    def __get_step_results(self):
        # get the step result for our agent
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        # get the camera observations
        imageData = decision_steps.obs[0][0,:,:,:]
        # receive the robot pose data via the string channel
        addi_info = self.env_channel.received_info
        # clear the received info, very important to do so; otherwise mess up the received info
        self.env_channel.clear_received_info()
        poseData = [float(x) for x in addi_info[-1].split()]

        # the orientation of the coordinate sys in Unity and here is different such that the sum of two is phase 90 degree
        poseData[-1] = 90 - poseData[-1]

        return imageData, poseData


    # This function actually actuates the agent/robot in the virtual environment.
    #
    # actuatorCommand:  the command that is used in the actuation process
    #
    def actuateRobot(self,actuatorCommand):
        # if the actuator command has more than 2 array entries, this is a teleport command, and will cause a teleport jump of the agent/robot (no physics support)
        if actuatorCommand.shape[0]>2:

            # call the teleportation routine
            [imageData, poseData]=self.step_simulation_without_physics(actuatorCommand[0],actuatorCommand[1],actuatorCommand[2])

            # return the data acquired from the robot/agent/environment
            return imageData, poseData

        else:
            # otherwise, this is a standard motion command with physics support (the agent/robot approaches the goal by actuating the robot's/agent's wheels)
            [imageData, poseData]=self.stepSimulation(actuatorCommand[0],actuatorCommand[1])

            # return the data acquired from the robot/agent/environment
            return imageData, poseData


    # This function teleports the robot to a novel pose.
    #
    # pose:     the pose to teleport to, format: [X,Z,Yaw]
    def setXYYaw(self, pose):
        action = ActionTuple(pose)
        self.env.set_actions(self.behavior_name, action)
        # forward the simulation by a tick (and execute action)
        self.env.step()

    # This function returns the limits of the environmental perimeter.
    def getLimits(self):
        # rearrange the elements for topology module to use
        world_limits = [[self.world_limits[0], self.world_limits[2]], [self.world_limits[1], self.world_limits[3]]]
        return np.asarray(world_limits)

    # This function returns the environmental perimeter by means of wall vertices/segments.
    def getWallGraph(self):
        return self.walls_limits, self.perimeter_nodes

    # This function shuts down Unity.
    def stopUnity(self):
        self.env.close()