import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import matplotlib as mat
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import Counter
# set the font size of the plots
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
mat.rc('font', **font)
# %%
agents = ['EC', 'DQN', 'DQN_online']
agent_colors = ['#2ca02c', '#ff7f0e', '#1f77b4'] # so green, orange, blue
envs = ['TMaze', 'TMaze_LV1', 'TMaze_LV2', 'DoubleTMaze', 'TunnelMaze_New']
envs_names = ['T-maze', 'long T-maze', 'H-maze', 'double T-maze', 'tunnel maze']
start_goal_nodes = {'TMaze': [[3], [11]], 'TMaze_LV1': [[5], [15]], 'TMaze_LV2': [[8], [21]],
                    'DoubleTMaze': [[15], [32]], 'TunnelMaze_New': [[44], [101]]}
epsilon=0.1
# get the path of this script
current_path = os.path.dirname(os.path.abspath(__file__))
################## visualize average learning curves #################
def flat_data(data, N=3):
    flattened_data = []
    for i in range(len(data)-N):
        flattened_data.append(np.mean(data[i:i+N]))
    return np.asarray(flattened_data)
epochs=range(1, 26)
for running_env, name in zip(envs, envs_names):
    # for performing statistical test
    samples = dict()
    fig3, axs3=plt.subplots(figsize=(7.5, 6))
    for agent, color in zip(agents, agent_colors):
        data_matrix = []
        for epoch in epochs:
            data_path = current_path + '/individuals/training/%s_%s_training_trajs_%s_%s.pickle' % (running_env, agent, epsilon, epoch)
            with open(data_path, 'rb') as handle:
                data = pickle.load(handle)
            num_steps = []
            for item in data:
                num_steps.append(len(item))
            data_matrix.append(num_steps)
        data_matrix = np.asarray(data_matrix)
        mu = flat_data(data_matrix.mean(axis=0))
        std = flat_data(data_matrix.std(axis=0))
        axs3.plot(mu, '-', label=agent, color=color,linewidth=3)
        nums=np.arange(1, len(mu)+1)
        samples[agent] = data_matrix
        # axs3.fill_between(nums, mu+std, mu-std, alpha=0.4, color=color)
    axs3.legend(fontsize=20)
    axs3.set_xlim([-2, 200])
    axs3.set_ylim([-10, 700])
    axs3.grid(True)
    axs3.set_title(name, fontsize=20)
    axs3.set_ylabel('# of time steps', fontsize=22)
    axs3.set_xlabel('trials', fontsize=22)
    plt.tight_layout()

################# average steps for the final solutions ##################
final_num_steps = []
training_stage = 500
for (agent, color) in zip(agents, agent_colors):
    final_num_steps.append([])
    for running_env in envs:
        final_num_steps[-1].append([])
        data_matrix = None
        data_dir=current_path+'/individuals/test/%s_%s_test_trajs_%s.pickle' % (running_env, agent, epsilon)
        data=pickle.load(open(data_dir, "rb"))
        for item in data[training_stage]:
            trial_length = len(item)
            final_num_steps[-1][-1].append(trial_length)

fig5, axs5 = plt.subplots(figsize=(14, 7))
w = 0.15
x = np.arange(1, len(envs)+1)
linewidth = 2
scatter_x = np.tile(x, (len(final_num_steps[0][0]),1))
for (i, fns) in enumerate(final_num_steps):
    axs5.bar(x+i*w, height=[np.mean(a) for a in fns], linewidth=linewidth,
             yerr=[np.std(a)/np.sqrt(np.size(a)) for a in fns], color=(0,0,0,0),
             width=w, edgecolor=agent_colors[i], align='center', capsize=12, label=agents[i])
    ## make the size of each dot proportional to the frequency of the data
    # count the occurrences of each point
    sizes=[]
    for item in fns:
        c = Counter(item)
        # create a list of the sizes, here multiplied by 10 for scale
        s = [c[(y)] for y in item]
        sizes.append(s)
    sizes = np.array(sizes)*10
    axs5.scatter(scatter_x.T+i*w, fns, color='black', s=sizes)
axs5.legend()
axs5.set_ylabel('# steps in test trial')
axs5.set_xticks(x+i*w/2)
axs5.set_ylim([0, 30])
axs5.set_title('%s' % training_stage)
axs5.set_xticklabels(envs_names)
axs5.grid()

# %%
# ######## visualize the trajectory that the agent takes in the test trial ###########
### load the topology of the env
running_env = envs[4]
cmaps = ['Greens', 'Oranges', 'Blues']
path = current_path + '/../environments/offline_unity/%s_Top.pickle' % running_env
with open(path, 'rb') as handle:
    data = pickle.load(handle)
# here each node is a XY coordinate and each edge is a list containing 2 integers, indicating a connectivity
# between 2 nodes
nodes = data['nodes']
edges = data['edges']
walls = data['walls']
# function for drawing the env topology
def draw_envs(axs):
    # draw the walls
    for w in walls:
        xy = w[:2]
        wh = w[2:] - w[:2]
        rect = mat.patches.Rectangle(xy, wh[0], wh[1], edgecolor='k', facecolor='none')
        axs.add_patch(rect)
    # draw the edges
    for e in edges:
        axs.plot(nodes[e][:, 0], nodes[e][:, 1], color='k', zorder=1)
    # draw the nodes
    axs.scatter(nodes[:, 0], nodes[:, 1], facecolors='white', edgecolors='b', s=80, zorder=2)
    # color the start and end nodes
    special = start_goal_nodes[running_env]
    axs.scatter(nodes[special[0][0]][0], nodes[special[0][0]][1], color='green', s=70, zorder=2)
    axs.scatter(nodes[special[1][0]][0], nodes[special[1][0]][1], color='red', s=70, zorder=2)

# function for defining arrow pos and orientations
def arrow_info(traj):
    # draw the arrows that define the trajectory
    arrow_pos = []
    # the X and Y components for each arrow direction
    U = []
    V = []
    for n in traj:
        # define the start and end of the arrow
        arrow_pos.append(nodes[n[0]])
        orientation = n[1]
        if orientation == 0:
            U.append(1)
            V.append(0)
        elif orientation == 90:
            U.append(0)
            V.append(1)
        elif orientation == -90:
            U.append(0)
            V.append(-1)
        elif orientation == -180:
            U.append(-1)
            V.append(0)

    arrow_pos = np.array(arrow_pos)
    return arrow_pos, U, V

def intersect2D(a, b):
  """
  Find row intersection between 2D numpy arrays, a and b.
  Returns another numpy array with shared rows
  """
  return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])

####### visualize the final solutions
epochs = np.arange(1, 3)
training_stage = 500
for (agent, agent_color) in zip(agents, cmaps):
    path = current_path + '/individuals/test/%s_%s_test_trajs_%s.pickle' % (running_env, agent, epsilon)
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
        for epoch in epochs:
            traj = data[training_stage][epoch]
            fig, axs = plt.subplots(figsize=[7.5, 9])
            draw_envs(axs)
            arrow_pos, U, V = arrow_info(traj)
            # the color of the arrows are defined w.r.t. its time step
            time_step = np.arange(1, len(traj)+1)
            axs.quiver(arrow_pos[:, 0], arrow_pos[:, 1], U, V, time_step, cmap=agent_color,
                          norm=colors.LogNorm(vmin=time_step.min()/20.0,vmax=time_step.max()*2.0))
            axs.set_title('TunnelMaze, %s' % agent)
            axs.set_xticks([])
            axs.set_yticks([])

# ######visualize the state the agent has visited and the spreading of the solution
epochs = np.arange(1, 26)
explore_rates = []
sol_spreading_rates = []
solution_ratio = []    # the percentage of # solution states to total # states
visited_ns_ratio = []  # the percentage of # visited non-solution states to total # states
non_visited_ratio = [] # the percentage of # non-visited states to total # states
training_stage = 500
for (agent, agent_color) in zip(agents, agent_colors):
    explore_rates.append([])
    sol_spreading_rates.append([])
    solution_ratio.append([])
    visited_ns_ratio.append([])
    non_visited_ratio.append([])
    for running_env in envs:
        explore_rates[-1].append([])
        sol_spreading_rates[-1].append([])
        solution_ratio[-1].append([])
        visited_ns_ratio[-1].append([])
        non_visited_ratio[-1].append([])
        ## load the data recording reward states
        path = current_path+'/individuals/test/%s_%s_reward_prop_%s.pickle' % (running_env, agent, epsilon)
        with open(path, 'rb') as handle:
            reward_prop_data = pickle.load(handle)
        # extract data run by run
        for epoch in epochs:
            #load the states that the agent has visited
            path = current_path + '/individuals/training/%s_%s_training_trajs_%s_%s.pickle' % (running_env, agent, epsilon, epoch)
            with open(path, 'rb') as handle:
                training_trajs = pickle.load(handle)
            visited_states = []
            for x in training_trajs:
                visited_states.extend(x)
            # extract unique states
            visited_states = np.unique(visited_states, axis=0)

            # extract the states where the agent can find the goal by starting on it for each run
            orientations = [0, 90, -180, -90]
            spreaded_sol = []
            for (i, node) in enumerate(reward_prop_data[training_stage][epoch-1]):
                for (j ,ori) in enumerate(node):
                    if ori:
                        spreaded_sol.append([i, orientations[j]])

            # find the state that the agent has visited and can find the solution from it
            spreaded_sol = intersect2D(visited_states, spreaded_sol)
            total_nodes = len(reward_prop_data[training_stage][epoch-1])*len(orientations)
            solution_ratio[-1][-1].append(len(spreaded_sol) / total_nodes)
            visited_ns_ratio[-1][-1].append((len(visited_states) - len(spreaded_sol)) / total_nodes)
            non_visited_ratio[-1][-1].append((total_nodes - len(visited_states)) / total_nodes)
            # how much env the agent has explored
            explore_rates[-1][-1].append(len(visited_states) / total_nodes)
            # calculate the rate of the number of sol states to the num of visited states
            sol_spreading_rates[-1][-1].append(len(spreaded_sol)/len(visited_states))

            ## start plotting example figures of reward propagation
            # fig, axs = plt.subplots(figsize=[7.5, 9])
            # draw_envs(axs)
            # # draw blackarrows on if the state has been visited
            # arrow_pos, U, V = arrow_info(visited_states)
            # for i in range(len(U)):
            #     axs.quiver(arrow_pos[i, 0], arrow_pos[i, 1], U[i], V[i], color='gray', scale=1, units='xy')
            # # draw colored arrows if the agent can find a solution on this state and ever visited the state
            # arrow_pos, U, V = arrow_info(spreaded_sol)
            # for i in range(len(U)):
            #     axs.quiver(arrow_pos[i, 0], arrow_pos[i, 1], U[i], V[i], color=agent_color, scale=1, units='xy')
            # axs.set_title('%s, %s' % (running_env, agent))
            # axs.set_xticks([])
            # axs.set_yticks([])

########## draw the solution spreading rate
w = 0.22
x = np.arange(1, len(envs)+1)
linewidth = 2
fs = [15, 7]
fig3, axs3 = plt.subplots(figsize=fs)
scatter_x = np.tile(x, (len(sol_spreading_rates[0][0]),1))
for (i, ssr) in enumerate(sol_spreading_rates):
    axs3.bar(x+i*w, height=[np.mean(a) for a in ssr], linewidth=linewidth,
             yerr=[np.std(a)/np.sqrt(np.size(a)) for a in ssr], color=(0,0,0,0),
             width=w, edgecolor=agent_colors[i], align='center', capsize=12, label=agents[i])
    axs3.scatter(scatter_x.T+i*w, ssr, color='black', s=20)
axs3.legend()
axs3.set_ylabel('# solution states to # explored states')
axs3.set_title('%s' % training_stage)
axs3.set_xticks(x+i*w/2)
axs3.set_xticklabels(envs_names)
axs3.grid()
#### draw the stacked bar for different kinds of states
fs = [15, 7]
fig5, axs5 = plt.subplots(figsize=fs)
for (i, (sr, vnr, nvr)) in enumerate(zip(solution_ratio, visited_ns_ratio, non_visited_ratio)):
    axs5.bar(x+i*w, height=[np.mean(a) for a in sr], linewidth=linewidth, color=agent_colors[i],
             width=0.9*w, edgecolor=(0,0,0,0), align='center', capsize=12)

    axs5.bar(x+i*w, height=[np.mean(a) for a in vnr], linewidth=linewidth, color='gray',
             bottom=[np.mean(a) for a in sr], width=0.9*w, edgecolor=(0,0,0,0), align='center', capsize=12)

    axs5.bar(x+i*w, height=[np.mean(a) for a in nvr], linewidth=linewidth, color=(0,0,0,0),
             bottom=[np.mean(a) for a in (np.array(sr)+np.array(vnr))], width=0.9*w, edgecolor=(0,0,0,0),
             align='center', capsize=12)
    # add an edge for the entire bar
    axs5.bar(x+i*w, height=np.ones(len(sr)), linewidth=linewidth, color=(0,0,0,0), width=0.9*w,
             edgecolor=agent_colors[i], align='center', capsize=12, label=agents[i])

axs5.legend(framealpha=1.0)
axs5.set_xticks(x+i*w/2)
axs5.set_title('%s' % training_stage)
axs5.set_xticklabels(envs_names)
axs5.set_ylabel('Fraction values')
axs5.grid()


plt.show()
