# basic imports
import os
import numpy as np
import PyQt5 as qt
import pyqtgraph as pg
import pickle
from tensorflow.compat.v1.keras import backend
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Convolution2D
# framework imports
from spatial_representations.topology_graphs.four_connected_graph_rotation import Four_Connected_Graph_Rotation
from observations.image_observations import ImageObservationUnity
from analysis.rl_monitoring.rl_performance_monitors import UnityPerformanceMonitor
# custom imports
from interfaces.oai_gym_interface import OAIGymInterface
from frontends.frontends_unity import FrontendUnityOfflineInterface
from agents.dqn_agent import DQNAgentBaseline
from agents.em_control import EM_control

# set some python environment properties
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # reduces the amount of debug messages from tensorflow.
backend.set_image_data_format(data_format='channels_last')
# whether visualize the performance monitor or not
visualOutput = False

def rewardCallback(values):
    reward = 0.0
    stopEpisode = False
    if values['currentNode'].goalNode:
        reward = 1.0
        stopEpisode = True

    return reward, stopEpisode

def test_rewardCallback(values):
    test_steps = 100
    elaped_steps = values['episode_step']
    reward=0.0
    stopEpisode=False

    if values['currentNode'].goalNode:
        reward=1.0
        stopEpisode=True

    # if the agent goes back to a same state from previous steps, the policy cannot be right, end episode
    episode_traj = np.array(values['episode_traj'])
    repeatation = np.where(np.prod(episode_traj == episode_traj[-1], axis=1))[0]
    if len(repeatation) >= 2:
        stopEpisode=True

    if elaped_steps == test_steps:
        stopEpisode=True

    return [reward,stopEpisode]

def trialEndCallback(trial, rlAgent, logs):
    if visualOutput:
        # update the visual elements if required
        rlAgent.performanceMonitor.update(trial, logs)
        if hasattr(qt.QtGui, 'QApplication'):
            qt.QtGui.QApplication.instance().processEvents()
        else:
            qt.QtWidgets.QApplication.instance().processEvents()

def singleRun(agent, running_env, data_folder, params, epoch, step_size=1.0, ifstore=True):
    # this is the main window for visual outputs
    mainWindow = None
    # if visual output is required, activate an output window
    if visualOutput:
        mainWindow = pg.GraphicsWindow(title="Performance monitor")
        layout = pg.GraphicsLayout(border=(30, 30, 30))
        mainWindow.setCentralItem(layout)
    # a global variable containing the start, goal location; will be defined later
    global start_goal_nodes
    # determine world info file path
    worldInfo = os.path.dirname(os.path.abspath(__file__)) + '/../../environments_unity/offline_unity/%s_ss%s_Infos.pickle' % \
                (running_env, step_size)
    # a dictionary that contains all employed modules
    modules = dict()
    modules['world'] = FrontendUnityOfflineInterface(worldInfo)
    modules['observation'] = ImageObservationUnity(modules['world'], mainWindow, visualOutput)
    modules['spatial_representation'] = Four_Connected_Graph_Rotation(modules, {'startNodes': start_goal_nodes[running_env][0],
                                        'goalNodes': start_goal_nodes[running_env][1], 'start_ori': 90, 'cliqueSize': 4}, step_size=step_size)
    modules['spatial_representation'].set_visual_debugging(visualOutput, mainWindow)
    modules['rl_interface'] = OAIGymInterface(modules, visualOutput, rewardCallback)

    learning_rate = 0.0001
    if agent == 'DQN_original':
        # this is the original DQN
        rlAgent = DQNAgentBaseline(modules, batch_size=params['batch_size'],
                                   epsilon=params['epsilon'], memoryCapacity=params['memory_capacity'], lr=learning_rate,
                                   trialBeginFcn=None, trialEndFcn=trialEndCallback)
    elif agent == 'DQN_sparse_replay':
        # this DQN replay "batch_size" of past experiences every "batch_size" step; for compensation, we increase the learning rate by batch_size
        rlAgent = DQNAgentBaseline(modules, batch_size=params['batch_size'], sparse_sample=True,
                                   epsilon=params['epsilon'], memoryCapacity=params['memory_capacity'], lr=learning_rate,
                                   trialBeginFcn=None, trialEndFcn=trialEndCallback)
    elif agent == 'DQN_single_replay':
        # this DQN replay 1 experience from the memory every step
        rlAgent = DQNAgentBaseline(modules, batch_size=1, epsilon=params['epsilon'],
                                   memoryCapacity=params['memory_capacity'], lr=learning_rate,
                                   trialBeginFcn=None, trialEndFcn=trialEndCallback)
    elif agent == 'DQN_online':
        # this DQN learns with online experiences
        rlAgent = DQNAgentBaseline(modules, batch_size=1, epsilon=params['epsilon'],
                                   memoryCapacity=0, lr=learning_rate, trialBeginFcn=None, trialEndFcn=trialEndCallback)
    elif agent == 'EC':
        rlAgent = EM_control(modules, epsilon=params['epsilon'], memoryCapacity=params['memory_capacity'],
                             trialBeginFcn=None, trialEndFcn=trialEndCallback)
    # set the performance monitor
    perfMon = UnityPerformanceMonitor(rlAgent, mainWindow, visualOutput)
    rlAgent.performanceMonitor = perfMon
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rlAgent = rlAgent
    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rlAgent = rlAgent
    ##########################################################
    # start the training
    # dist for storing the reward propagation information
    reward_prop_run = dict()
    # dist for storing training and test trajs
    trajs_run = dict()
    for m in range(len(params['numberOfTrials']) - 1):
        # training
        rlAgent.train(params['numberOfTrials'][m+1] - params['numberOfTrials'][m], params['maxNumberOfSteps'])
        ### after the training, place the agent on each (node, orientation) and see if it can find the goal
        # change the reward function
        modules['rl_interface'].rewardCallback = test_rewardCallback
        num_nodes = modules['spatial_representation'].get_node_number()
        orientations = [0, 90, -180, -90]
        # list for storing the reward propagation information
        reward_prop = []
        for j in range(num_nodes):
            reward_prop.append([])
            for ori in orientations:
                modules['spatial_representation'].reset_start_nodes([j], ori)
                # perform a test trial
                history = rlAgent.predict(1)
                # delete the trajactory of this test trial
                del modules['spatial_representation'].trajectories[-1]
                if history.history['episode_reward'][0] > 0:
                    reward_prop[-1].append(True)
                elif history.history['episode_reward'][0] == 0:
                    reward_prop[-1].append(False)

        reward_prop_run[params['numberOfTrials'][m + 1]] = reward_prop
        # change the reward function back to the training one
        modules['rl_interface'].rewardCallback = rewardCallback
        # set the starting node back to the original point
        modules['spatial_representation'].reset_start_nodes(start_goal_nodes[running_env][0], 90)

    trajs_run['training'] = modules['spatial_representation'].trajectories
    # after the training, test the agent for one trial
    rlAgent.predict(1, params['maxNumberOfSteps'])
    trajs_run['test'] = modules['spatial_representation'].trajectories[-1]

    if ifstore:
        # we store the entire trajectory history of the agent
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        data_dir = data_folder + '/TrainingTrajs_%s_%s_%s.pickle' % (agent, running_env, epoch)
        with open(data_dir, 'wb') as handle:
            pickle.dump(trajs_run, handle)
        # we also store the reward propagation information
        data_dir = data_folder + '/RewardPropa_%s_%s_%s.pickle' % (agent, running_env, epoch)
        with open(data_dir, 'wb') as handle:
            pickle.dump(reward_prop_run, handle)

    # end the training
    backend.clear_session()
    modules['world'].stopUnity()
    if visualOutput:
        mainWindow.close()

## main scripts
global start_goal_nodes
envs = ['TunnelMaze_LV1', 'TunnelMaze_LV2', 'TunnelMaze_LV3', 'TunnelMaze_LV4']
start_goal_nodes = {'TunnelMaze_LV1': [[1], [14]], 'TunnelMaze_LV2': [[6], [32]], 'TunnelMaze_LV3': [[21], [60]],
                        'TunnelMaze_LV4': [[42], [94]]}
agents = ['EC', 'DQN_original', 'DQN_online', 'DQN_sparse_replay', 'DQN_single_replay']

params = {'numberOfTrials': [0, 250, 500], 'maxNumberOfSteps': 600, 'memory_capacity': 200000, 'batch_size': 32, 'epsilon': 0.1}
epochs = 50

data_folder = os.path.dirname(os.path.abspath(__file__)) + '/../../data/learning_compare'
for running_env in envs:
    for agent in agents[1:]:
        for epoch in range(epochs):
            singleRun(agent, running_env, data_folder, params, epoch, ifstore=True)
