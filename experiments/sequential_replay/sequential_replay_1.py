# basic imports
import os
import numpy as np
import random
import PyQt5 as qt
import pyqtgraph as pg
import pickle
from tensorflow.compat.v1.keras import backend
import tensorflow as tf
# framework imports
from spatial_representations.topology_graphs.four_connected_graph_rotation import Four_Connected_Graph_Rotation
from observations.image_observations import ImageObservationUnity
from analysis.rl_monitoring.rl_performance_monitors import UnityPerformanceMonitor
from interfaces.oai_gym_interface import OAIGymInterface
from frontends.frontends_unity import FrontendUnityOfflineInterface
from memory_modules.sfma_memory import SFMAMemory
from memory_modules.memory_utils.metrics import Learnable_DR
from agents.deep_sfma import SFMA_DQNAgent
from analysis.sequence_analyzer import sequence_analyze

# set some python environment properties
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # reduces the amount of debug messages from tensorflow.
backend.set_image_data_format(data_format='channels_last')
# configuration for GPU computing
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.compat.v1.Session(config=config)
backend.set_session(sess)

visualOutput = False
# reward function
def rewardCallback(values):
    # the standard reward for each step taken is negative, making the agent seek short routes
    reward = 0.0
    stopEpisode = False
    if values['currentNode'].goalNode:
        reward = 1.0
        stopEpisode = True

    return reward, stopEpisode

def trialEndCallback(trial, rlAgent, logs):
    if visualOutput:
        # update the visual elements if required
        rlAgent.performanceMonitor.update(trial, logs)
        if hasattr(qt.QtGui, 'QApplication'):
            qt.QtGui.QApplication.instance().processEvents()
        else:
            qt.QtWidgets.QApplication.instance().processEvents()

data_folder = os.path.dirname(os.path.abspath(__file__)) + '/../../data/sequential_replay_1'
def single_run(running_env, replay_type, num_replay, step_size=1.0, beta=1.0, batch_size=32, epoch=0):
    #### choose and load the environment
    start_goal_nodes = {'TunnelMaze_LV1': [[1], [14]], 'TunnelMaze_LV2': [[6], [32]], 'TunnelMaze_LV3': [[21], [60]],
                        'TunnelMaze_LV4': [[42], [94]], 'TunnelMaze_LV4_hc': [[204], [458]]}
    # this is the main window for visual outputs
    mainWindow=None
    # if visual output is required, activate an output window
    if visualOutput:
        mainWindow = pg.GraphicsWindow(title="Unity Environment Plot")
        layout = pg.GraphicsLayout(border=(30, 30, 30))
        mainWindow.setCentralItem(layout)

    # determine world info file path
    worldInfo = os.path.dirname(os.path.abspath(__file__)) + '/../../environments_unity/offline_unity/%s_ss%s_Infos.pickle' % (running_env, step_size)
    # a dictionary that contains all employed modules
    modules = dict()
    modules['world'] = FrontendUnityOfflineInterface(worldInfo)
    modules['observation'] = ImageObservationUnity(modules['world'], mainWindow, visualOutput)
    modules['spatial_representation'] = Four_Connected_Graph_Rotation(modules, {'startNodes':start_goal_nodes[running_env][0], 'goalNodes':start_goal_nodes[running_env][1], 'start_ori': 90, 'cliqueSize':4}, step_size=step_size)
    modules['spatial_representation'].set_visual_debugging(visualOutput, mainWindow)
    modules['rl_interface']=OAIGymInterface(modules, visualOutput, rewardCallback, withStateIdx=True)

    #### load the memory replay module, one is SMA and the other Random
    numberOfActions = modules['rl_interface'].action_space.n
    numberOfStates = modules['world'].numOfStates()
    gammaDR = 0.2
    ## for the SMA, load the DR matrix for the env
    ## if there is no stored DR, start a simple simulation where an agent randomly explore the env and update DR incrementally, then store it
    DR_metric = Learnable_DR(numberOfStates, gammaDR)
    DR_path = os.path.dirname(os.path.abspath(__file__)) + '/../../memory_modules/stored/DR_%s_ss%s_gamma_%s.pickle' % (running_env, step_size, gammaDR)
    ifPretrained = DR_metric.loadD(DR_path)
    if not ifPretrained:
        modules['spatial_representation'].generate_behavior_from_action('reset')
        for _ in range(200000):
            currentStateIdx = modules['world'].envData['imageIdx']
            action = np.random.randint(low=0, high=numberOfActions)
            modules['spatial_representation'].generate_behavior_from_action(action)
            nextStateIdx = modules['world'].envData['imageIdx']
            # update DR matrix with one-step experience
            DR_metric.updateD(currentStateIdx, nextStateIdx, lr=0.1)
        # store the learned DR for this env
        DR_metric.storeD(DR_path)
    # initialize hippocampal memory
    HM = SFMAMemory(numberOfStates, numberOfActions, DR_metric)
    # load the agent
    epsilon = 0.1
    rlAgent = SFMA_DQNAgent(modules, HM, replay_type, epsilon, 0.95, with_replay=True, online_learning=False,
                            trialBeginFcn=None, trialEndFcn=trialEndCallback)
    # number of online replays starting from the terminal state
    rlAgent.online_replays_per_trial = num_replay[0]
    # number of offline replays
    rlAgent.offline_replays_per_trial = num_replay[1]
    rlAgent.batch_size = batch_size

    # common settings
    rlAgent.memory.beta = beta
    rlAgent.memory.mode = 'reverse'
    rlAgent.memory.reward_mod = True
    rlAgent.memory.reward_modulation = 1.0
    rlAgent.logging_settings['steps'] = True
    rlAgent.logging_settings['replay_traces'] = True

    # set the performance monitor
    perfMon=UnityPerformanceMonitor(rlAgent,mainWindow,visualOutput)
    rlAgent.performanceMonitor=perfMon
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rlAgent=rlAgent
    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rlAgent=rlAgent
    # start the training
    rlAgent.train(numberOfTrials=500, maxNumberOfSteps=600)
    # end the training
    backend.clear_session()
    modules['world'].stopUnity()
    if mainWindow is not None:
        mainWindow.close()
    # extract replayed state index in each batch
    replayed_history = []
    for batch in rlAgent.logs['replay_traces']['end']:
        statebatch = [[e['state'], e['next_state']] for e in batch]
        replayed_history.append(statebatch)

    ifanalyze = True
    if ifanalyze:
    	data_folder = data_folder + '/beta_%s/%s+%s' % (beta, num_replay[0], num_replay[1])
		if not os.path.exists(data_folder):
			os.makedirs(data_folder)
        # store the trajectories in all training trials
        data_path = data_folder + '/TrainingTrajs_%s_%s_%s.pickle' % (replay_type, running_env, epoch)
        with open(data_path, 'wb') as handle:
            pickle.dump(modules['spatial_representation'].trajectories, handle)

        data_path = data_folder + '/ReplayBatches_%s_%s_%s.pickle' % (replay_type, running_env, epoch)
        with open(data_path, 'wb') as handle:
            pickle.dump(replayed_history, handle)

if __name__ == "__main__":
    running_env = 'TunnelMaze_LV4'
    step_size = 1.0
    betas = [0, 0.0001, 0.001, 0.01, 0.1, 1, 2, 5, 10]
    epochs = range(50)
    for i in epochs:
        for beta in betas:
            single_run(running_env, 'SR_AU', step_size=step_size, batch_size=32, num_replay=[10, 0], beta=beta, epoch=i)
            single_run(running_env, 'SR_AU', step_size=step_size, batch_size=32, num_replay=[20, 0], beta=beta, epoch=i)
            single_run(running_env, 'SR_AU', step_size=step_size, batch_size=32, num_replay=[50, 0], beta=beta, epoch=i)
