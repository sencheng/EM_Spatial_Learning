import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import matplotlib as mat
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import Counter
# set the font size of the plots
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 22}
mat.rc('font', **font)
running_env = 'TunnelMaze_New'
epsilon=0.1
# get the path of this script
current_path = os.path.dirname(os.path.abspath("__file__"))
print(current_path)

def dot_freq_size(data, scale):
    ## make the size of each dot proportional to the frequency of the data
    sizes=[]
    for item in data:
        # count the occurrences of each point
        c = Counter(item)
        # create a list of the sizes, here multiplied by 10 for scale
        s = [c[(y)] for y in item]
        sizes.append(s)
    sizes = np.array(sizes)*scale
    return sizes

########### compare learning curves of hybrid agent with and without replay
agents = ['Hybrid_max', 'Hybrid_max_nr']
agent_names = ['Hybrid', 'Hybrid_no_replay']
agent_colors = ['white', 'black']
epochs=range(1, 101)
training_stage = 200
samples = dict()
fig3, axs3=plt.subplots(figsize=(8,6))
for agent, color, name in zip(agents, agent_colors, agent_names):
    data_matrix = []
    for epoch in epochs:
        data_path = current_path + '/hybrid/%s/%s_%s_steps_%s_%s.pickle' % (training_stage, running_env, agent, epsilon, epoch)
        with open(data_path, 'rb') as handle:
            data = pickle.load(handle)
        num_steps = []
        for item in data:
            num_steps.append(item['nb_episode_steps'])
        data_matrix.append(num_steps)
    data_matrix = np.asarray(data_matrix)
    mu = data_matrix.mean(axis=0)
    se = data_matrix.std(axis=0) / np.sqrt(data_matrix.shape[0])
    nums=np.arange(1, len(mu)+1)
    axs3.plot(nums, mu, '-', marker='o', markerfacecolor=color, markeredgecolor='k',
              label=name, linewidth=2, color='k', markersize=8)
    if name == 'Hybrid_no_replay':
        se = [np.zeros(se.shape), se]
    elif name == 'Hybrid':
        se = [se, np.zeros(se.shape)]
    axs3.errorbar(x=nums, y=mu, yerr=se, color='k', capsize=3, linewidth=2)
    samples[agent] = data_matrix
# axs3.legend(fontsize=20)
axs3.set_xlim([1, 30])
axs3.set_ylim([10, 1000])
axs3.set_yscale('log')
axs3.yaxis.set_major_formatter(mat.ticker.ScalarFormatter())
axs3.grid(True)
axs3.set_ylabel('# of time steps', fontsize=22)
axs3.set_xlabel('trials', fontsize=22)
plt.tight_layout()

# %%
################## # of steps in the test trial for each component of hybrid agent
components = ['EC', 'NN']
com_colors = ['#2ca02c', '#ff7f0e']
training_stages = np.array([10, 20, 30, 50, 100, 200])
fig4, axs4=plt.subplots(figsize=(15,8))
w = 0.2
x = np.arange(1, len(training_stages)+1)
scatter_x = np.tile(x, (100,1))
linewidth = 2.5
for i, agent in enumerate(agents):
    data_dir=current_path+'/hybrid/%s_%s_all_numsteps_%s.pickle' % (running_env, agent, epsilon)
    data=pickle.load(open(data_dir, "rb"))
    for j, com in enumerate(components):
        num_steps = np.array(data[com])
        mu = np.mean(num_steps, axis=0)
        se = np.std(num_steps, axis=0) / np.sqrt((num_steps.shape[0]))
        # draw the bar plots for each component
        if agent == 'Hybrid_max':
            color = [0,0,0,0]
        else:
            color = com_colors[j]
        axs4.bar(x+i*w/2+j*w, height=mu, linewidth=linewidth, color=color,
                 width=w/2, edgecolor=com_colors[j], align='center', capsize=12, label=com + ' component', zorder=1)
        dot_sizes = np.transpose(dot_freq_size(np.transpose(num_steps), 2))
        axs4.scatter(scatter_x.T+i*w/2+j*w, num_steps, color='black', s=dot_sizes, zorder=2)

    # draw for the complete hybrid agent
    num_steps = samples[agent][:, training_stages-1]
    mu = np.mean(num_steps, axis=0)
    se = np.std(num_steps, axis=0) / np.sqrt((num_steps.shape[0]))
    # draw the bar plots for each component
    j+=1
    if agent == 'Hybrid_max':
        color = [0,0,0,0]
    else:
        color = 'r'
    axs4.bar(x+i*w/2+j*w, height=mu, linewidth=linewidth, color=color,
             width=w/2, edgecolor='r', align='center', capsize=12, label='complete agent', zorder=1)
    ## make the size of each dot proportional to the frequency of the data
    dot_sizes = np.transpose(dot_freq_size(np.transpose(num_steps), 2))
    axs4.scatter(scatter_x.T+i*w/2+j*w, num_steps, color='black', s=dot_sizes, zorder=2)

axs4.set_xticks(x+i*w/2)
axs4.set_xticklabels(training_stages)
axs4.grid(True)
axs4.set_xlabel('trial')
axs4.set_ylabel('# of time steps')

plt.show()
