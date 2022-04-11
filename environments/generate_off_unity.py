import pyqtgraph as pg
import pickle
import os
from agents.dqn_agents import DQNAgentBaseline
from frontends.frontends_unity import FrontendUnityInterface
from spatial_representations.topology_graphs.four_connected_graph_rotation import Four_Connected_Graph_Rotation
from observations.image_observations import ImageObservationBaseline
from interfaces.oai_gym_interface import OAIGymInterface
from analysis.rl_monitoring.rl_performance_monitors import RLPerformanceMonitorBaseline

visualOutput = True
def rewardCallback(values):
    reward=0.0
    stopEpisode=False
    return [reward,stopEpisode]

def trialBeginCallback(trial,rlAgent):
    pass

def trialEndCallback(trial,rlAgent,logs):
    if visualOutput:
        # update the visual elements if required
        rlAgent.interfaceOAI.modules['spatial_representation'].updateVisualElements()
        rlAgent.performanceMonitor.update(trial,logs)

# The length of the edge in the topology graph
step_size = 1.0
use_grayscale = False
running_env = 'TMaze'
# this is the main window for visual outputs
mainWindow=None
# if visual output is required, activate an output window
if visualOutput:
    mainWindow = pg.GraphicsWindow(title="Unity Environment Plot")
    layout = pg.GraphicsLayout(border=(30, 30, 30))
    mainWindow.setCentralItem(layout)
# a dictionary that contains all employed modules
modules=dict()
modules['world']=FrontendUnityInterface(running_env)
modules['observation']=ImageObservationBaseline(modules['world'],mainWindow,visualOutput, use_grayscale)
modules['spatial_representation']=Four_Connected_Graph_Rotation(modules, {'startNodes':[0],'goalNodes':[1], 'start_ori': 90, 'cliqueSize':4}, step_size=step_size)
modules['spatial_representation'].set_visual_debugging(visualOutput,mainWindow)
modules['rl_interface']=OAIGymInterface(modules,visualOutput,rewardCallback)
rlAgent = DQNAgentBaseline(interfaceOAI=modules['rl_interface'], enable_CNN=False, memory_type='sequential', use_random_projections=True,
                            epsilon=0.1, memoryCapacity=5000, trialBeginFcn=trialBeginCallback, trialEndFcn=trialEndCallback)
# set the experimental parameters
perfMon=RLPerformanceMonitorBaseline(rlAgent,mainWindow,visualOutput)
rlAgent.performanceMonitor=perfMon
# eventually, allow the OAI class to access the robotic agent class
modules['rl_interface'].rlAgent=rlAgent
# and allow the topology class to access the rlAgent
modules['spatial_representation'].rlAgent=rlAgent

############## store the environment images ##########################
# spawn the agent on every node and every direction
num_nodes = modules['spatial_representation'].get_node_number()
env_info = {}
orientations = [0, 90, -180, -90]
for k in range(num_nodes):
    # set start node and orientation
    for j in range(len(orientations)):
        modules['spatial_representation'].reset_start_nodes([k], orientations[j])
        # get the images for this node, orientation
        observation = modules['rl_interface'].reset()
        # store it in the dict
        key = str([k, orientations[j]])
        env_info[key] = observation

# also store the topology parameters for generating topology graph
env_info['world_limits'] = modules['world'].world_limits
env_info['walls_limits'] = modules['world'].walls_limits
env_info['perimeter_nodes'] = modules['world'].perimeter_nodes
# storing...
current_path = os.path.dirname(os.path.abspath(__file__))
store_path = '/../environments/offline_unity/%s_ss%s_Infos.pickle' % (running_env, step_size)
data_dir=current_path+store_path
with open(data_dir, "wb") as data_harbour:
    pickle.dump(env_info, data_harbour)
