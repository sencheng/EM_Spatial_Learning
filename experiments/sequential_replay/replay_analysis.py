import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
from collections import Counter
import matplotlib.colors as colors
import matplotlib.cm
import matplotlib as mat
import pickle, os
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.collections import LineCollection
## set the font size of the plots
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 22}
mat.rc('font', **font)
# useful functions
def inputs_sim(X, Y):
    '''
    A measurement of how similar two lists X and Y is.
    X and Y should have the same length and both contain elements of (stateIdx, nextstateIdx)
    If the stateIdx is the same, we think the inputs from X and Y are the same
    '''
    m = 0.0
    ifpaired = [False for _ in Y]
    for i in range(len(X)):
        for j in range(len(Y)):
            if ifpaired[j]: continue
            if X[i][0]==Y[j][0] and X[i][1]==Y[j][1]:
                m += 1
                ifpaired[j] = True
                break
    return m/len(X)
def extract_batch(batch):
    extracted = []
    for e in batch['reverse']: extracted.extend(e)
    for e in batch['forward']: extracted.extend(e)
    extracted.extend(batch['single'])
    return extracted
def compute_sim(image_souce, method):
    shape = image_souce.shape
    if method == 'euclidean':
        sim_mat = np.zeros((shape[0], shape[0]))
        for i in range(shape[0]):
            for j in range(shape[0]):
                sim_mat[i, j] = np.mean(1-np.sqrt(np.sum((image_souce[i]-image_souce[j])**2, axis=2)/3))
    elif method == 'cosine':
        vectors = np.reshape(image_souce, (shape[0], np.prod(shape[1:])))
        sim_mat = cosine_similarity(vectors)
    elif method == 'corr':
        vectors = np.reshape(image_souce, (shape[0], np.prod(shape[1:])))
        sim_mat = np.corrcoef(vectors)

    return sim_mat
def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

start_goal_nodes = {'TunnelMaze_LV1': [[1], [14]], 'TunnelMaze_LV2': [[6], [32]], 'TunnelMaze_LV3': [[21], [60]],
                    'TunnelMaze_LV4': [[42], [94]]}

data_folder = os.path.dirname(os.path.abspath(__file__)) + '/../../data/sequential_replay_1'
project_folder = os.path.dirname(os.path.abspath(__file__)) + '/../..'
running_env = 'TunnelMaze_LV4'
batch_size = 32

## load the environment data
scenarioPath = os.path.dirname(os.path.abspath(__file__)) + \
            '/../../environments_unity/offline_unity/%s_ss%s_Infos.pickle' % \
            (running_env, 1.0)
with open(scenarioPath, 'rb') as handle:
    world_info = pickle.load(handle)
obs_keys = list(world_info.keys())[:-3]

#### Do sequential batches indeed have more serial correlations? ####
ifdraw1 = False
if ifdraw1:
    # parameters
    betas = [2, 5]
    colors = ['gray', 'black']
    num_replays = [10, 20]
    epochs = range(50)
    num_trial = 100
    # initialize the plot
    fig, axs = plt.subplots(figsize=(6, 5))
    w = 0.2
    xs = np.arange(1, len(num_replays)+1)
    # for sequential replay
    m = 0
    r_type = 'SR_AU'
    for beta in betas:
        for num_replay in num_replays:
            batch_sim = np.zeros((num_replay, num_replay))
            for epoch in epochs:
                ## load the entire replay batches
                # sequential batch
                batch_path = data_folder + '/beta_%s/%s+%s/ReplayBatches_%s_%s_%s.pickle' % (
                    beta, num_replay, 0, r_type, running_env, epoch)
                with open(batch_path, 'rb') as handle:
                    batches = pickle.load(handle)
                for t in range(num_trial):
                    temp = np.zeros((num_replay, num_replay))
                    for i in range(num_replay):
                        batch1 = batches[t*num_replay+i]
                        for j in range(i, num_replay):
                            batch2 = batches[t*num_replay+j]
                            temp[i][j] = inputs_sim(batch1, batch2)
                            temp[j][i] = inputs_sim(batch2, batch1)
                    batch_sim += temp

    fig, axes = plt.subplots(1, 2, tight_layout=True, facecolor='white')
    keys = ['sequential', 'random']
    for i, key in enumerate(keys):
        sim_mat = np.zeros((batch_size, batch_size))
        batches = batch_data[key]
        for batch in batches:
            states = [world_info[obs_keys[e[0]]] for e in batch]
            sim_mat += compute_sim(np.asarray(states), 'euclidean')
        sim_mat /= len(batches)
        im = axes[i].imshow(sim_mat, vmin=0.7, vmax=1.0)
        axes[i].title.set_text('Replay batch similarity, %s, %s' % (running_env, key))
    plt.colorbar(im)
    plt.show()

###### how useful are the batches sampled at each trial ######
ifdraw2 = False
if ifdraw2:
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

    r_types = ['SR_AU', 'RR_AU']
    num_replays = [50]
    betas = [1]
    epochs = range(50)
    transited_edges = {}
    replayed_rotations = {}
    num_trial = 40

    # first extract the numbers of times each node transition and rotation
    max_times_translation = 0
    max_times_rotation = 0
    # for drawing the environment topology
    top_path = project_folder + '/environments_unity/offline_unity/%s_ss1.0_Top.pickle' % running_env

    for r_type in r_types:
        for beta in betas:
            if r_type == 'RR_AU': beta = 2
            for num_replay in num_replays:
                print(r_type, beta, num_replay)
                transited_edges['%s, %s, %s'%(r_type, beta, num_replay)] = {}
                replayed_rotations['%s, %s, %s'%(r_type, beta, num_replay)] = {}
                for epoch in epochs:
                    ## load the entire replay batches
                    # sequential batch
                    batch_path = data_folder + '/beta_%s/%s+%s/ReplayBatches_%s_%s_%s.pickle' % (
                    beta, num_replay, 0, r_type, running_env, epoch)
                    with open(batch_path, 'rb') as handle:
                        batches = pickle.load(handle)

                    for batch in batches[:(num_trial*num_replay)]:
                        # find out transitions of nodes
                        for e in batch:
                            state = obs_keys[e[0]].replace('[','').replace(']','').split(',')
                            next_state = obs_keys[e[1]].replace('[','').replace(']','').split(',')

                            # if the node idexs are different, an translation is replayed
                            if state[0] != next_state[0]:
                                # if this transition is seen first time
                                if '%s,%s' % (state[0], next_state[0]) not in list(transited_edges['%s, %s, %s'%(r_type, beta, num_replay)].keys()):
                                    transited_edges['%s, %s, %s'%(r_type, beta, num_replay)]['%s,%s' % (state[0], next_state[0])] = 1
                                else:
                                    transited_edges['%s, %s, %s'%(r_type, beta, num_replay)]['%s,%s' % (state[0], next_state[0])] += 1
                            else:   # otherwise a rotation or standing still is replayed
                                # if this rotation is seen first time
                                if '%s' % state[0] not in list(replayed_rotations['%s, %s, %s'%(r_type, beta, num_replay)].keys()):
                                    replayed_rotations['%s, %s, %s'%(r_type, beta, num_replay)]['%s' % state[0]] = 1
                                else:
                                    replayed_rotations['%s, %s, %s'%(r_type, beta, num_replay)]['%s' % state[0]] += 1

                max_times_translation = max(max_times_translation, max(list(transited_edges['%s, %s, %s'%(r_type, beta, num_replay)].values())))
                max_times_rotation = max(max_times_rotation, max(list(replayed_rotations['%s, %s, %s'%(r_type, beta, num_replay)].values())))

    max_times_translation /= len(epochs)
    max_times_rotation /= len(epochs)
    print(max_times_translation, max_times_rotation)
    max_times_translation = 850
    max_times_rotation = 3900

    for r_type in r_types:

        for beta in betas:
            if r_type == 'RR_AU': beta = 2
            for num_replay in num_replays:
                print('Plotting')
                print(r_type, beta, num_replay)
                fig, axs = plt.subplots(figsize=(7, 8))
                nodes = draw_envs(top_path, axs)
                for key in list(transited_edges['%s, %s, %s'%(r_type, beta, num_replay)].keys()):
                    freq = transited_edges['%s, %s, %s'%(r_type, beta, num_replay)][key] / (max_times_translation * len(epochs))
                    edge = [int(x) for x in key.split(',')]
                    axs.plot(nodes[edge][:, 0], nodes[edge][:, 1], color='red', zorder=2, lw=10.0, alpha=freq)

                for key in list(replayed_rotations['%s, %s, %s'%(r_type, beta, num_replay)].keys()):
                    freq = replayed_rotations['%s, %s, %s'%(r_type, beta, num_replay)][key] / (max_times_rotation * len(epochs))
                    node = int(key)
                    axs.scatter(x=nodes[node][0], y=nodes[node][1], color='blue', zorder=3, s=70.0, alpha=freq)

                plt.title('%s, num_replay: %s, Mbeta: %s, num_trials: %s' % (r_type, num_replay, beta, num_trial), fontsize=18)
    plt.show()

## within each update, how are the batches different from each other?
ifdraw3 = True
if ifdraw3:
    # parameters
    betas = [0, 1, 2, 5, 10]
    # colors = ['gray', 'black']
    colors = ['blue', 'yellow', 'red']
    patterns = ['o', 'v', 's']
    num_replays = [10, 20, 50]
    epochs = range(2)
    num_trial = 200
    # initialize the plot
    fig, axs = plt.subplots(figsize=(9, 8))
    w = 0.1
    xs = np.arange(1, len(betas)+1)

    m = 0
    for num_replay in num_replays:
        batch_sim_avg = []
        # for sequential replay
        for beta in betas:
            r_type = 'SR_AU'
            batch_sim = np.zeros((num_replay, num_replay))
            for epoch in epochs:
                ## load the entire replay batches
                # sequential batch
                batch_path = data_folder + '/beta_%s/%s+%s/ReplayBatches_%s_%s_%s.pickle' % (
                    beta, num_replay, 0, r_type, running_env, epoch)
                with open(batch_path, 'rb') as handle:
                    batches = pickle.load(handle)
                for t in range(num_trial):
                    temp = np.zeros((num_replay, num_replay))
                    for i in range(num_replay):
                        batch1 = batches[t*num_replay+i]
                        for j in range(i, num_replay):
                            batch2 = batches[t*num_replay+j]
                            temp[i][j] = inputs_sim(batch1, batch2)
                            temp[j][i] = inputs_sim(batch2, batch1)
                    batch_sim += temp

            batch_sim /= num_trial*len(epochs)
            # compute the average sim
            average = (np.sum(batch_sim)-num_replay)/(num_replay**2-num_replay) # so we exclude the diagonal element
            batch_sim_avg.append(average)
        # draw bar plots
        # axs.bar(xs+m*w, batch_sim_avg, width=w, color='white', edgecolor='black', hatch=patterns[m], label='beta=%s'%beta)
        axs.plot(betas, batch_sim_avg, marker=patterns[m], linewidth=3, label='# replays=%s'%num_replay, markersize=15, color=colors[m])
        m += 1

    # for random replay we draw a dashed line
    r_type = 'RR_AU'
    beta = 2 # not useful, just for finding the folder
    batch_sim = np.zeros((num_replay, num_replay))
    for epoch in epochs:
        ## load the entire replay batches
        # sequential batch
        batch_path = data_folder + '/beta_%s/%s+%s/ReplayBatches_%s_%s_%s.pickle' % (
            beta, num_replay, 0, r_type, running_env, epoch)
        with open(batch_path, 'rb') as handle:
            batches = pickle.load(handle)
        for t in range(num_trial):
            temp = np.zeros((num_replay, num_replay))
            for i in range(num_replay):
                batch1 = batches[t*num_replay+i]
                for j in range(i, num_replay):
                    batch2 = batches[t*num_replay+j]
                    temp[i][j] = inputs_sim(batch1, batch2)
                    temp[j][i] = inputs_sim(batch2, batch1)
            batch_sim += temp
    batch_sim /= num_trial*len(epochs)
    # compute the average sim
    average = (np.sum(batch_sim)-num_replay)/(num_replay**2-num_replay) # so we exclude the diagonal element
    axs.axhline(average, linestyle='dashed', linewidth=1.5, color='k', label='random replay (DQN)')
    axs.set_xlabel('Beta')
    axs.set_ylabel('Event-level similarity')
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()

### draw the environment similarity matrix ####
ifdraw4 = False
if ifdraw4:
    fig, ax = plt.subplots(tight_layout=True, facecolor='white')
    states = list(world_info.values())[:-3]
    sim_mat = compute_sim(np.asarray(states), 'euclidean')
    im = ax.imshow(sim_mat, vmin=0.7, vmax=1.0)
    ax.title.set_text('Inputs similarity, %s ' % running_env)
    plt.colorbar(im)
    plt.show()
