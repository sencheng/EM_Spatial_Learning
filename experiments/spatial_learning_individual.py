import numpy as np
import pyqtgraph as pg
import pickle
import os
import tensorflow as tf
from tensorflow.compat.v1.keras import backend
from agents.dqn_baseline import DQNAgentBaseline
from agents.em_control import EM_control
from frontends.frontends_unity_offline import FrontendUnityInterface
from spatial_representations.topology_graphs.four_connected_graph_rotation import Four_Connected_Graph_Rotation
from observations.image_observations import ImageObservationBaseline
from interfaces.oai_gym_interface import OAIGymInterface
from analysis.rl_monitoring.rl_performance_monitors import RLPerformanceMonitorBaseline
import matplotlib.pyplot as plt

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
num_episodes = [0, 250, 500]
num_runs = 25
use_grayscale = False
envs = ['TMaze', 'TMaze_LV1', 'TMaze_LV2', 'DoubleTMaze', 'TunnelMaze_New']
start_goal_nodes = {'TMaze': [[3], [11]], 'TMaze_LV1': [[5], [15]], 'TMaze_LV2': [[8], [21]],
                    'DoubleTMaze': [[15], [32]], 'TunnelMaze_New': [[44], [101]]}
agents = ['EC', 'DQN', 'DQN_online']
params = {'memory_capacity': 200000, 'batch_size':32, 'epsilon': 0.1}


current_path = os.path.dirname(os.path.abspath(__file__))

for running_env in envs:
    for agent in agents:
        # dict for storing the trajectory in the test trial
        test_trajs = dict()
        # list for storing the reward propagation information
        reward_prop_runs = dict()
        for num_ep in num_episodes[1:]:
            test_trajs[num_ep] = []
            reward_prop_runs[num_ep] = []

        for i in range(num_runs):
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
            ## start the training
            for m in range(len(num_episodes)-1):
                # training
                rlAgent.train(num_episodes[m+1]-num_episodes[m])
                # perform a test trial
                rlAgent.predict(1)
                # record the trajectory that the agent took in the test trial, and delete it from the buffer
                test_trajs[num_episodes[m+1]].append(modules['spatial_representation'].trajectories[-1])
                del modules['spatial_representation'].trajectories[-1]
                del rlAgent.engagedCallbacks.logs_data[-1]

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
                        del rlAgent.engagedCallbacks.logs_data[-1]
                        if history.history['episode_reward'][0] > 0:
                            reward_prop[-1].append(True)
                        elif history.history['episode_reward'][0] == 0:
                            reward_prop[-1].append(False)
                reward_prop_runs[num_episodes[m+1]].append(reward_prop)
                # change the reward function back to the training one
                modules['rl_interface'].rewardCallback = rewardCallback
                # set the starting node back to the original point
                modules['spatial_representation'].reset_start_nodes(start_goal_nodes[running_env][0], 90)

            ## store the trajectory that the agent take in each training trial
            store_path = '/../data/individuals/training/%s_%s_training_trajs_%s_%s.pickle' % (running_env, agent, params['epsilon'], i+1)
            agent_data_dir = current_path+store_path
            with open(agent_data_dir, 'wb') as handle:
                pickle.dump(modules['spatial_representation'].trajectories, handle)

            # one run is over
            backend.clear_session()

        # store the trajectory that the agent followed in the test trials for all the runs
        store_path = '/../data/individuals/test/%s_%s_test_trajs_%s.pickle' % (running_env, agent, params['epsilon'])
        agent_data_dir = current_path+store_path
        with open(agent_data_dir, 'wb') as handle:
            pickle.dump(test_trajs, handle)

        # store the reward propagation for all the runs
        store_path = '/../data/individuals/test/%s_%s_reward_prop_%s.pickle' % (running_env, agent, params['epsilon'])
        agent_data_dir = current_path+store_path
        with open(agent_data_dir, 'wb') as handle:
            pickle.dump(reward_prop_runs, handle)
