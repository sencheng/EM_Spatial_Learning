import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
from collections import Counter
import matplotlib.colors as colors
import matplotlib.cm

## set the font size of the plots
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 20}
mat.rc('font', **font)
## env and agent information
envs = ['TunnelMaze_LV1', 'TunnelMaze_LV2','TunnelMaze_LV3','TunnelMaze_LV4']
start_goal_nodes = {'TunnelMaze_LV1': [[1], [14]], 'TunnelMaze_LV2': [[6], [32]], 'TunnelMaze_LV3': [[21], [60]],
                        'TunnelMaze_LV4': [[42], [94]]}
agents = ['EC', 'DQN_original', 'DQN_online', 'DQN_sparse_replay', 'DQN_single_replay']
names = ['EC', 'DQN', 'DQN_online', 'DQN_sparse_sampling', 'DQN_batch_size_1']
agent_colors = ['green', 'orange', 'blue', 'yellow', 'black']
# get the path of data storage
data_folder = os.path.dirname(os.path.abspath(__file__)) + '/../../data/learning_compare'
project_folder = os.path.dirname(os.path.abspath(__file__)) + '/../..'
epochs=range(50)
def flat_data(data, N=3):
    flattened_data = []
    for i in range(len(data)-N):
        flattened_data.append(np.mean(data[i:i+N]))
    return np.asarray(flattened_data)
################## visualize average learning curves ###################
ifdraw = True
if ifdraw:
    test_steps = {}
    for running_env in envs:
        fig1, axs1=plt.subplots(figsize=(7.5, 6))
        # for EC and DQN online
        for agent, color, name in zip(agents[1:], agent_colors[1:], names[1:]):
            data_matrix = []
            test_steps['%s_%s' % (running_env, agent)] = []
            for epoch in epochs:
                data_path = data_folder + '/TrainingTrajs_%s_%s_%s.pickle' % (agent, running_env, epoch)
                with open(data_path, 'rb') as handle:
                    data = pickle.load(handle)
                training_trajs = data['training']
                # test_steps['%s_%s' % (running_env, agent)].append(len(data['test']))
                num_steps = []
                for item in training_trajs:
                    num_steps.append(len(item))
                data_matrix.append(num_steps)
            data_matrix = np.asarray(data_matrix)
            mu = flat_data(data_matrix.mean(axis=0))
            std = flat_data(data_matrix.std(axis=0))
            trials = np.arange(1, len(mu) + 1)
            axs1.plot(mu, '-', label=name, color=color,linewidth=2.5)
            # axs1.fill_between(trials, mu+std, mu-std, alpha=0.3, color=color)

        axs1.legend(fontsize=16)
        axs1.set_xlim([-2, 550])
        axs1.set_ylim([-10, 600])
        axs1.grid(True)
        axs1.set_title(running_env, fontsize=20)
        axs1.set_ylabel('# of time steps', fontsize=20)
        axs1.set_xlabel('trials', fontsize=20)
        plt.tight_layout()

    # ################# average steps for the test trials ##################
    final_num_steps = []
    for key in list(test_steps.keys()):
        test_steps[key] = np.asarray(test_steps[key])

    fig2, axs2 = plt.subplots(figsize=(14, 7))
    w = 0.15
    x = np.arange(1, len(envs)+1)
    linewidth = 2
    scatter_x = np.tile(x, (len(final_num_steps[0][0]),1))
    for (i, fns) in enumerate(final_num_steps):
        label = agents[i]
        if label == 'DQN_original': label = 'DQN'
        axs2.bar(x+i*w, height=[np.mean(a) for a in fns], linewidth=linewidth,
                 yerr=[np.std(a)/np.sqrt(np.size(a)) for a in fns], color=(0,0,0,0),
                 width=w, edgecolor=agent_colors[i], align='center', capsize=12, label=label)
        ## make the size of each dot proportional to the frequency of the data
        # count the occurrences of each point
        sizes=[]
        for item in fns:
            c = Counter(item)
            # create a list of the sizes, here multiplied by 10 for scale
            s = [c[(y)] for y in item]
            sizes.append(s)
        sizes = np.array(sizes)*10
        axs2.scatter(scatter_x.T+i*w, fns, color='black', s=sizes)
    axs2.legend(fontsize=16)
    axs2.set_title('Asymptotic performance')
    axs2.set_ylabel('# time steps')
    axs2.set_xticks(x+i*w/2)
    axs2.set_ylim([0, 50])
    axs2.set_xticklabels(envs, fontsize=18)
    axs2.grid()
    plt.show()

# function for drawing the env topology
def draw_envs(env_top_path, axs):
    with open(env_top_path, 'rb') as handle:
        data = pickle.load(handle)
    # here each node is a XY coordinate and each edge is a list containing 2 integers, indicating a connectivity
    # between 2 nodes
    nodes = data['nodes']
    edges = data['edges']
    walls = data['walls']
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
    axs.scatter(nodes[:, 0], nodes[:, 1], facecolors='white', edgecolors='b', s=80, zorder=3)
    # color the start and end nodes
    special = start_goal_nodes[running_env]
    axs.scatter(nodes[special[0][0]][0], nodes[special[0][0]][1], color='green', s=70, zorder=3)
    axs.scatter(nodes[special[1][0]][0], nodes[special[1][0]][1], color='red', s=70, zorder=3)
    return nodes
# function for defining arrow pos and orientations
def arrow_info(traj, nodes):
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
### load the topology of the env
running_env = 'TunnelMaze_LV4'
top_path = project_folder+'/environments_unity/offline_unity/%s_ss1.0_Top.pickle' % running_env
###################### visualize trajectories ########################
ifdraw1 = False
if ifdraw1:
    agents = ['EC', 'DQN_original', 'DQN_online']
    cmaps = ['Greens', 'Oranges', 'Blues']
    epochs = np.arange(50)
    trial_index = 499   # which trial to visualize
    for (agent, agent_color) in zip(agents[1:2], cmaps[1:2]):
        for epoch in epochs:
            data_path = data_folder + '/TrainingTrajs_%s_%s_%s.pickle' % (agent, running_env, epoch)
            with open(data_path, 'rb') as handle:
                data = pickle.load(handle)
            traj = data[trial_index]
            fig3, axs3 = plt.subplots(figsize=(7, 8))
            nodes = draw_envs(top_path, axs3)
            arrow_pos, U, V = arrow_info(traj, nodes)
            # the color of the arrows are defined w.r.t. its time step
            color_steps = np.linspace(0.5, 1, len(traj))
            cmap = matplotlib.cm.get_cmap(agent_color)
            # define the size of the arrow
            width = 0.06
            length = 0.5
            for xy, dx, dy, t in zip(arrow_pos, U, V, color_steps):
                axs3.arrow(xy[0], xy[1], dx*length, dy*length, width=width, head_width=4*width,
                           head_length=3.5*width, color=cmap(t),zorder=2)
            axs3.set_title('%s, %s' % (running_env, agent))
            axs3.set_xticks([])
            axs3.set_yticks([])
            # plt.savefig(project_folder+'/images/traj_%s_trial%s.svg' % (agent, trial_index), format='svg')
    plt.show()

##################### reward propagation ################################
ifdraw2 = True
if ifdraw2:
    epochs = np.arange(0,50)
    num_reward_prop = 0
    explore_rates = []
    sol_spreading_rates = []
    solution_ratio = []    # the percentage of # solution states to total # states
    visited_ns_ratio = []  # the percentage of # visited non-solution states to total # states
    non_visited_ratio = [] # the percentage of # non-visited states to total # states
    training_stage = 250
    for (agent, agent_color) in zip(agents[0:3], agent_colors[0:3]):
        explore_rates.append([])
        sol_spreading_rates.append([])
        solution_ratio.append([])
        visited_ns_ratio.append([])
        non_visited_ratio.append([])
        for running_env in envs[0:]:
            explore_rates[-1].append([])
            sol_spreading_rates[-1].append([])
            solution_ratio[-1].append([])
            visited_ns_ratio[-1].append([])
            non_visited_ratio[-1].append([])
            ## load the data recording reward states
            m = 0
            for epoch in epochs:
                ## load the data recording reward states
                path = data_folder + '/RewardPropa_%s_%s_%s.pickle' % (agent, running_env, epoch)
                with open(path, 'rb') as handle:
                    reward_prop_data = pickle.load(handle)
                #load the states that the agent has visited
                path = data_folder + '/TrainingTrajs_%s_%s_%s.pickle' % (agent, running_env, epoch)
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
                for (i, node) in enumerate(reward_prop_data[training_stage]):
                    for (j ,ori) in enumerate(node):
                        if ori:
                            spreaded_sol.append([i, orientations[j]])

                # find the state that the agent has visited and can find the solution from it
                spreaded_sol = intersect2D(visited_states, spreaded_sol)
                total_nodes = len(reward_prop_data[training_stage])*len(orientations)
                solution_ratio[-1][-1].append(len(spreaded_sol) / total_nodes)
                visited_ns_ratio[-1][-1].append((len(visited_states) - len(spreaded_sol)) / total_nodes)
                non_visited_ratio[-1][-1].append((total_nodes - len(visited_states)) / total_nodes)
                # how much env the agent has explored
                explore_rates[-1][-1].append(len(visited_states) / total_nodes)
                # calculate the rate of the number of sol states to the num of visited states
                sol_spreading_rates[-1][-1].append(len(spreaded_sol)/len(visited_states))

                if m < num_reward_prop:
                    m += 1
                    ## start plotting example figures of reward propagation
                    fig, axs = plt.subplots(figsize=[7.5, 9])
                    nodes = draw_envs(top_path, axs)
                    # draw blackarrows on if the state has been visited
                    arrow_pos, U, V = arrow_info(visited_states, nodes)
                    for i in range(len(U)):
                        axs.quiver(arrow_pos[i, 0], arrow_pos[i, 1], U[i], V[i], color='gray', scale=1, units='xy')
                    # draw colored arrows if the agent can find a solution on this state and ever visited the state
                    arrow_pos, U, V = arrow_info(spreaded_sol, nodes)
                    for i in range(len(U)):
                        axs.quiver(arrow_pos[i, 0], arrow_pos[i, 1], U[i], V[i], color=agent_color, scale=1, units='xy')
                    axs.set_title('%s, %s' % (running_env, agent))
                    axs.set_xticks([])
                    axs.set_yticks([])
                    plt.show()

    ########## draw the solution spreading rate
    w = 0.22
    x = np.arange(1, len(envs)+1)
    linewidth = 2
    fs = [15, 7]
    fig3, axs3 = plt.subplots(figsize=fs)
    scatter_x = np.tile(x, (len(sol_spreading_rates[0][0]),1))
    for (i, ssr) in enumerate(sol_spreading_rates):
        label = agents[i]
        if label == 'DQN_original': label = 'DQN'
        axs3.bar(x+i*w, height=[np.mean(a) for a in ssr], linewidth=linewidth,
                 yerr=[np.std(a)/np.sqrt(np.size(a)) for a in ssr], color=(0,0,0,0),
                 width=w, edgecolor=agent_colors[i], align='center', capsize=12, label=label)
        ## make the size of each dot proportional to the frequency of the data
        # count the occurrences of each point
        sizes=[]
        for item in ssr:                                                                          
            c = Counter(item)
            # create a list of the sizes, here multiplied by 10 for scale
            s = [c[(y)] for y in item]
            sizes.append(s)
        sizes = np.transpose(np.array(sizes)*5)
        # print(scatter_x.shape, sizes.shape)
        axs3.scatter(scatter_x.T+i*w, ssr, color='black', s=sizes)
    axs3.legend()
    axs3.set_ylabel('# solution states to # explored states')
    axs3.set_title('%s' % training_stage)
    axs3.set_xticks(x+i*w/2)
    axs3.set_xticklabels(envs)
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

    # axs5.legend(framealpha=1.0)
    axs5.set_xticks(x+i*w/2)
    axs5.set_title('%s' % training_stage)
    axs5.set_xticklabels(envs)
    axs5.set_ylabel('Fraction values')
    axs5.grid()
plt.show()









