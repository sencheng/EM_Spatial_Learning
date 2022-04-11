import numpy as np
import pyqtgraph as pg
import pickle
import os
import tensorflow as tf
from tensorflow.compat.v1.keras import backend
from agents.dqn_baseline import DQNAgentBaseline
from agents.em_control import EM_control
from agents.hybrid_agent import HybridAgent
from frontends.frontends_unity_offline import FrontendUnityInterface
from spatial_representations.topology_graphs.four_connected_graph_rotation import Four_Connected_Graph_Rotation
from observations.image_observations import ImageObservationBaseline
from interfaces.oai_gym_interface import OAIGymInterface
from analysis.rl_monitoring.rl_performance_monitors import RLPerformanceMonitorBaseline
import matplotlib.pyplot as plt

# set some python environment properties
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # reduces the amount of debug messages from tensorflow.
backend.set_image_data_format(data_format='channels_last')
# configuration for GPU computing
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.compat.v1.Session(config=config)
backend.set_session(sess)

visualOutput = False
max_steps = 600
test_steps = 100

def rewardCallback(values):
    elaped_steps = values['episode_step']
    reward=0.0
    stopEpisode=False

    if values['currentNode'].goalNode:
        reward=1.0
        stopEpisode=True

    if elaped_steps == max_steps:
        stopEpisode=True

    return [reward,stopEpisode]

def test_rewardCallback(values):
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

def trialBeginCallback(trial,rlAgent):

    pass

def trialEndCallback(trial,rlAgent,logs):

    if visualOutput:
        # update the visual elements if required
        rlAgent.interfaceOAI.modules['spatial_representation'].updateVisualElements()
        rlAgent.performanceMonitor.update(trial,logs)

########################
num_episodes = [0, 10, 20, 30, 50, 100, 200]
num_runs = 100
use_grayscale = False
envs = ['TunnelMaze_New']
start_goal_nodes = {'TMaze': [[3], [11]], 'TMaze_LV1': [[5], [15]], 'TMaze_LV2': [[8], [21]],
                    'DoubleTMaze': [[15], [32]], 'TunnelMaze_New': [[44], [101]]}
agents = ['Hybrid_max', 'Hybrid_max_nr', 'DQN', 'EC']

params = {'memory_capacity': 20000, 'batch_size':32, 'epsilon': 0.1}

current_path = os.path.dirname(os.path.abspath(__file__))

for running_env in envs:
    for agent in agents:
        test_steps = {'EC':[], 'NN':[], 'Hybrid': []}
        for i in range(num_runs):
            for key in test_steps.keys():
                test_steps[key].append([])

            # this is the main window for visual outputs
            mainWindow=None
            # if visual output is required, activate an output window
            if visualOutput:
                mainWindow = pg.GraphicsWindow(title="Unity Environment Plot")
                layout = pg.GraphicsLayout(border=(30, 30, 30))
                mainWindow.setCentralItem(layout)
            topo_info = {'startNodes':start_goal_nodes[running_env][0],'goalNodes':start_goal_nodes[running_env][1], 'start_ori': 90, 'cliqueSize':4}
            # a dictionary that contains all employed modules
            modules=dict()
            modules['world']=FrontendUnityInterface(running_env)
            modules['observation']=ImageObservationBaseline(modules['world'],mainWindow,visualOutput, use_grayscale)
            modules['spatial_representation']=Four_Connected_Graph_Rotation(modules, topo_info)
            modules['spatial_representation'].set_visual_debugging(visualOutput,mainWindow)
            modules['rl_interface']=OAIGymInterface(modules,visualOutput,rewardCallback)
            if agent == 'DQN':
                rlAgent = DQNAgentBaseline(interfaceOAI=modules['rl_interface'], enable_CNN=True, memory_type='sequential', batch_size=params['batch_size'],
                                           epsilon=params['epsilon'], memoryCapacity=params['memory_capacity'], trialBeginFcn=trialBeginCallback, trialEndFcn=trialEndCallback)
            elif agent == 'EC':
                rlAgent = EM_control(interfaceOAI=modules['rl_interface'], epsilon=params['epsilon'], memoryCapacity=params['memory_capacity'], trialBeginFcn=trialBeginCallback, trialEndFcn=trialEndCallback)
            elif agent == 'DQN_online':
                rlAgent = DQNAgentBaseline(interfaceOAI=modules['rl_interface'], enable_CNN=True, memory_type='nomemory', batch_size=1,
                                           epsilon=params['epsilon'], memoryCapacity=params['memory_capacity'], trialBeginFcn=trialBeginCallback, trialEndFcn=trialEndCallback)
            elif agent == 'Hybrid_sum':
                rlAgent = HybridAgent(interfaceOAI=modules['rl_interface'], epsilon=params['epsilon'], memoryCapacity=params['memory_capacity'], hybrid_format='weighted_sum', sigma=0.5, with_replay=True,
                                      trialBeginFcn=trialBeginCallback, trialEndFcn=trialEndCallback)
            elif agent == 'Hybrid_sum_nr':
                rlAgent = HybridAgent(interfaceOAI=modules['rl_interface'], epsilon=params['epsilon'], memoryCapacity=params['memory_capacity'], hybrid_format='weighted_sum', sigma=0.5, with_replay=False,
                                      trialBeginFcn=trialBeginCallback, trialEndFcn=trialEndCallback)
            elif agent == 'Hybrid_max':
                rlAgent = HybridAgent(interfaceOAI=modules['rl_interface'], epsilon=params['epsilon'], memoryCapacity=params['memory_capacity'], hybrid_format='take_max', with_replay=True, trialBeginFcn=trialBeginCallback, trialEndFcn=trialEndCallback)
            elif agent == 'Hybrid_max_nr':
                rlAgent = HybridAgent(interfaceOAI=modules['rl_interface'], epsilon=params['epsilon'], memoryCapacity=params['memory_capacity'], hybrid_format='take_max', with_replay=False, trialBeginFcn=trialBeginCallback, trialEndFcn=trialEndCallback)

            # set the experimental parameters
            perfMon=RLPerformanceMonitorBaseline(rlAgent,mainWindow,visualOutput)
            rlAgent.performanceMonitor=perfMon
            # eventually, allow the OAI class to access the robotic agent class
            modules['rl_interface'].rlAgent=rlAgent
            # and allow the topology class to access the rlAgent
            modules['spatial_representation'].rlAgent=rlAgent

            for m in range(len(num_episodes)-1):
                # training
                rlAgent.train(num_episodes[m+1]-num_episodes[m])
                # store the number of trial steps for each training trial
                store_path = '/../data/hybrid/%s/%s_%s_steps_%s_%s.pickle' % (num_episodes[m+1], running_env, agent, params['epsilon'], i+1)
                agent_data_dir=current_path+store_path
                with open(agent_data_dir, "wb") as data_harbour:
                    pickle.dump(rlAgent.engagedCallbacks.logs_data, data_harbour)

                if agent == 'Hybrid_max' or agent == 'Hybrid_max_nr':
                    # store the number of steps that the agent select a EC q value or DQN q value
                    #store_path = '/../data/hybrid/num_steps/%s/%s_%s_selections_%s_%s.pickle' % (num_episodes[m+1], running_env, agent, params['epsilon'], i+1)
                    #agent_data_dir = current_path + store_path
                    #with open(agent_data_dir, "wb") as data_harbour:
                        #pickle.dump(rlAgent.select_history, data_harbour)

                    # at the same time, test if the DQN or EC in the hybrid agent alone can find the target
                    rlAgent.hybrid_format = 'weighted_sum'
                    # DQN
                    rlAgent.sigma = 0.0
                    rlAgent.predict(1)
                    test_steps['NN'][-1].append(len(modules['spatial_representation'].trajectories[-1]))
                    del modules['spatial_representation'].trajectories[-1]
                    del rlAgent.engagedCallbacks.logs_data[-1]
                    # EC
                    rlAgent.sigma = 1.0
                    rlAgent.predict(1)
                    test_steps['EC'][-1].append(len(modules['spatial_representation'].trajectories[-1]))
                    del modules['spatial_representation'].trajectories[-1]
                    del rlAgent.engagedCallbacks.logs_data[-1]
                    # set the hybrid mode to original
                    rlAgent.hybrid_format = 'take_max'
                    # check the hybrid agent's performace
                    rlAgent.predict(1)
                    test_steps['Hybrid'][-1].append(len(modules['spatial_representation'].trajectories[-1]))
                    del modules['spatial_representation'].trajectories[-1]
                    del rlAgent.engagedCallbacks.logs_data[-1]

            # one run is over
            backend.clear_session()

        # store the trajectory that the agent followed in the test trial in the end
        store_path = '/../data/hybrid/%s_%s_all_numsteps_%s.pickle' % (running_env, agent, params['epsilon'])
        agent_data_dir = current_path+store_path
        with open(agent_data_dir, 'wb') as handle:
            pickle.dump(test_steps, handle)
