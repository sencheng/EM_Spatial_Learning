import os, re
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
from collections import Counter
import matplotlib.colors as colors
import matplotlib.cm
from frontends.frontends_unity import FrontendUnityOfflineInterface
from spatial_representations.topology_graphs.four_connected_graph_rotation import Four_Connected_Graph_Rotation
from analysis.sequence_analyzer import sequence_analyze
## set the font size of the plots
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 22}
mat.rc('font', **font)
## env and agent information
start_goal_nodes = {'TunnelMaze_LV1': [[1], [14]], 'TunnelMaze_LV2': [[6], [32]], 'TunnelMaze_LV3': [[21], [60]],
                    'TunnelMaze_LV4': [[42], [94]], 'TunnelMaze_New': [[44], [101]], 'TMaze': [[3], [11]]}

project_folder = os.path.dirname(os.path.abspath(__file__)) + '/../..'
batch_size = 32

def flat_data(data, N=5):
    flattened_data = []
    for i in range(len(data)-N):
        flattened_data.append(np.mean(data[i:i+N]))
    return np.asarray(flattened_data)
# functions for visualizing env topology
def draw_envs(running_env, env_top_path, axs):
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
    axs.scatter(nodes[special[0][0]][0], nodes[special[0][0]][1], color='blue', s=70, zorder=3)
    axs.scatter(nodes[special[1][0]][0], nodes[special[1][0]][1], color='green', s=70, zorder=3)
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

def visualize_lc(data_folder, running_env, params, handles):
    extracted_num_steps = {}
    fig1, axs1 = handles
    for i, r_type in enumerate(params['replay_types']):
        data_matrix = []
        for epoch in params['epochs']:
            data_path = data_folder + '/TrainingTrajs_%s_%s_%s.pickle' % (r_type, running_env, epoch)
            with open(data_path, 'rb') as handle:
                training_trajs = pickle.load(handle)
            num_steps = []
            for item in training_trajs:
                num_steps.append(len(item))
            data_matrix.append(num_steps)
        data_matrix = np.asarray(data_matrix)
        extracted_num_steps['%s_%s'%(running_env, r_type)]=data_matrix
        mu = flat_data(data_matrix.mean(axis=0))
        std = flat_data(data_matrix.std(axis=0))
        trials = np.arange(1, len(mu) + 1)
        axs1.plot(mu, '-', color=params['replay_colors'][i], linewidth=2, linestyle=params['linestyles'][i])
        # axs1.legend(fontsize=16)
        axs1.set_xlim([-2, 500])
        axs1.set_ylim([-10, 600])
        axs1.grid(True)
        axs1.set_ylabel('# of time steps', fontsize=20)
        axs1.set_xlabel('trials', fontsize=20)
    plt.tight_layout()
    return extracted_num_steps

def sequence_len_dist(data_folder, running_env, r_type, epochs):
    num_epoch = len(epochs)
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 6))
    seq_len = {'forward':[], 'reverse':[]}
    for epoch in epochs:
        data_path = data_folder + '/ReplayBatches_%s_%s_%s.pickle' % (r_type, running_env, epoch)
        with open(data_path, 'rb') as handle:
            replayed_batches = pickle.load(handle)
        replayed_sqs = sequence_analyze(replayed_batches)
        for replay_batch in replayed_sqs:  # for each replay batch
            if len(replay_batch['forward']) > 0:
                 for x in replay_batch['forward']:
                     seq_len['forward'].append(len(x))
            if len(replay_batch['reverse']) > 0:
                for x in replay_batch['reverse']:
                    seq_len['reverse'].append(len(x))
            # finally, add single elements into both forward and reverse sequences
            if len(replay_batch['single']) > 0:
                for x in replay_batch['single']:
                    seq_len['forward'].append(1)
                    seq_len['reverse'].append(1)
    # draw histogram for the sequence length distribution, summing up all epochs
    forward_hist = axs2[0].hist(seq_len['forward'], bins=np.linspace(min(seq_len['forward']), max(seq_len['forward']) + 1,
                max(seq_len['forward']) - min(seq_len['forward']) + 2) - 0.5, density=True)
    reverse_hist = axs2[1].hist(seq_len['reverse'], bins=np.linspace(min(seq_len['reverse']), max(seq_len['reverse']) + 1,
                    max(seq_len['reverse']) - min(seq_len['reverse']) + 2) - 0.5, density=True)
    fig2.suptitle("Distribution of the sequence length, env: %s, r_type: %s" % (running_env, r_type), fontsize=15)
    axs2[0].set_xlabel('Length of the sequence')
    axs2[0].set_ylabel('Percentage')
    axs2[0].grid(True)
    axs2[1].set_xlabel('Length of the sequence')
    axs2[1].set_ylabel('Percentage')
    axs2[1].grid(True)
    axs2[0].set_title('# forward: %s' % int(len(seq_len['forward'])/num_epoch))
    axs2[1].set_title('# reverse: %s' % int(len(seq_len['reverse'])/num_epoch))
    plt.tight_layout()

    # plot how much elements each sequence bin contain
    fig3, axs3 = plt.subplots()
    values = list(reverse_hist[0])
    for _ in range(batch_size-len(values)):
        values.append(0.0)
    weighted_values = np.array([x*(i+1) for i, x in enumerate(values)])
    bins = np.arange(1, len(values)+1)
    axs3.bar(bins, weighted_values/np.sum(weighted_values))
    axs3.set_xticks(range(0, batch_size, 5))
    # axs3.set_title('Distribution for number of elements, env: %s, r_type: %s' % (running_env, r_type), fontsize=15)
    axs3.set_ylabel('Proportion')
    plt.tight_layout()

def plot_replay_sq(data_folder, running_env, r_type, epoch, num_seq, params):
    '''
    This function visualize the longest seqeunces
    '''
    modules = {}
    worldInfo = os.path.dirname(os.path.abspath(__file__)) + \
                '/../../environments_unity/offline_unity/%s_ss%s_Infos.pickle' % (running_env, 1.0)
    world_module = FrontendUnityOfflineInterface(worldInfo)
    modules['world'] = world_module
    topology_module = Four_Connected_Graph_Rotation(modules, {'startNodes': [0], 'goalNodes': [1], 'start_ori': 90,
                                                              'cliqueSize': 4}, step_size=1.0)
    top_path = project_folder + '/environments_unity/offline_unity/%s_ss%s_Top.pickle' % (running_env, 1.0)
    replay_path = data_folder + '/ReplayBatches_%s_%s_%s.pickle' % (r_type, running_env, epoch)
    with open(replay_path, 'rb') as handle:
        replayed_batches = pickle.load(handle)
    replayed_sqs = sequence_analyze(replayed_batches)
    env_keys = list(world_module.env.keys())
    reverse_sq = []
    for sq in replayed_sqs:
        reverse_sq.extend(sq['reverse'])
    reverse_sq_len = np.array([len(x) for x in reverse_sq])
    # extract all sequences which have the largest length
    longest_seq_idx = np.where(reverse_sq_len == np.max(reverse_sq_len))[0]
    # if the num_seq desired to be plotted is smaller than the total num of seqs,
    # then randomly select num_seq; otherwise plot all of the seqs.
    if len(longest_seq_idx) > num_seq:
        seq_to_plot = random.choices(longest_seq_idx, k=num_seq)
    else:
        seq_to_plot = longest_seq_idx
    for idx in seq_to_plot:
        # load the env topology and visualize it
        fig1, axs1 = plt.subplots(figsize=(8, 8))
        nodes_positions = draw_envs(running_env, top_path, axs1)

        example_seq = reverse_sq[idx]
        replay_sq = []
        for item in example_seq:
            state_top = env_keys[item[1]]
            state_top = re.sub("[\[\]]", "", state_top)
            state_top = state_top.split(',')
            replay_sq.append([int(state_top[0]), int(state_top[1])])
        arrow_pos, U, V = arrow_info(replay_sq, nodes_positions)
        color_steps = np.linspace(0, 1, len(arrow_pos))
        cmap = mat.cm.get_cmap('turbo')
        # define the size of the arrow
        width = 0.1
        length = 0.5
        for xy, dx, dy, t in zip(arrow_pos, U, V, color_steps):
            axs1.arrow(xy[0], xy[1], dx * length, dy * length, width=width, head_width=3.5 * width,
                       head_length=3 * width, color=cmap(t), zorder=2)
        axs1.set_title('Env: %s; R_type: %s; Epoch: %s' % (running_env, r_type, epoch))
        axs1.set_xticks([])
        axs1.set_yticks([])
    plt.show()

if __name__ == "__main__":
    # get the path of data storage
    data_folder = os.path.dirname(os.path.abspath(__file__)) + '/../../data/sequential_replay_1'
    envs = ['TunnelMaze_LV4']
    params = {'replay_colors': ['black', 'orange'], 'linestyles': ['solid', 'solid'],
              'replay_types': ['SR_AU', 'RR_AU'], 'epochs': range(50)}
    beta = 0.01
    num_replays = [10, 20, 50]
    # visulaize learning curves
    ifdraw1 = True
    if ifdraw1:
        for num_replay in num_replays:
            handles = plt.subplots()
            data_folder1 = data_folder + '/beta_%s/%s+%s' % (beta, num_replay, 0)
            visualize_lc(data_folder1, envs[0], params, handles)

    # visualize seq length distribution
    ifdraw2 = True
    if ifdraw2:
        num_replay = 20
        data_folder1 = data_folder + '/beta_%s/%s+%s' % (beta, num_replay, 0)
        sequence_len_dist(data_folder1, envs[0], params['replay_types'][1], params['epochs'])

    # visualize replayed sequences
    ifdraw3 = True
    if ifdraw3:
        num_replay = 20
        data_folder1 = data_folder + '/beta_%s/%s+%s' % (beta, num_replay, 0)
        plot_replay_sq(data_folder1, envs[0], 'SR_AU', epoch=12, num_seq=10, params=params)

    # compute the relative performance during the training
    ifdraw4 = True
    if ifdraw4:
        betas = [0, 0.0001, 0.001, 0.01, 0.1, 1, 2, 5, 10]
        num_replays = [10, 20, 50]
        colors = ['blue', 'yellow', 'red']
        markers = ['o', 'v', 's']
        r_types = ['SR_AU', 'RR_AU']
        epochs = range(50)
        fig, axs = plt.subplots(figsize=(9, 8))
        # coordinates = []
        for j, num_replay in enumerate(num_replays):
            relative_perfs = []  # each value represents the relative performance between SR_AU and RR_AU
            for beta in betas:
                # coordinates.append([beta, num_replay])
                avg_steps = np.zeros(2) # for sequential replay and random replay
                data_folder1 = data_folder + '/beta_%s/%s+%s' % (beta, num_replay, 0)
                for i, r_type in enumerate(r_types):
                    for epoch in epochs:
                        data_path = data_folder1 + '/TrainingTrajs_%s_%s_%s.pickle' % (r_type, envs[0], epoch)
                        with open(data_path, 'rb') as handle:
                            training_trajs = pickle.load(handle)
                        num_steps = []
                        for item in training_trajs:
                            num_steps.append(len(item))
                        avg_steps[i] += np.mean(num_steps)
                    avg_steps[i] /= len(epochs)
                relative_perfs.append(avg_steps[1]/avg_steps[0])
            # plot
            # for beta = 0, plot a single point
            axs.plot(betas[0], relative_perfs[0], marker=markers[j], color=colors[j], markersize=15)
            # for other betas, connect the data points
            axs.plot(betas[1:], relative_perfs[1:], linewidth=2.5, marker=markers[j], color=colors[j],
                     markersize=15, label='# replay=%s'%num_replay)
            axs.set_xlabel('Beta')
            axs.set_ylabel('Relative performance')

        axs.axhline(1.0, linestyle='dashed', linewidth=1.5, color='black')
        plt.legend(fontsize=20)
        plt.tight_layout()

    plt.show()
