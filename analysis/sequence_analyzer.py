# basic imports
import os
import re
import time 
import numpy as np
import matplotlib.pyplot as plt

def sequence_analyze(replay_history, fig_path=None):
    '''
    This function takes in a list of replay batches, and return another list where each batch has been
    divided into sequences in the state space.
    '''
    divided_history = []
    sequence_length = {'forward':[], 'reverse':[]}
    for batch in replay_history:
        divided_batch = {'forward':[], 'reverse':[], 'single':[]}
        detect_forward = False
        detect_reverse = False
        for i in range(len(batch)-1):
            if batch[i][0] == batch[i+1][1]:  # we detect a reverse sequence
                detect_forward = False
                if not detect_reverse:
                    detect_reverse = True # a reverse sequence storage starts
                    divided_batch['reverse'].append([batch[i], batch[i+1]])
                else:  # append the next element into this sequence
                    divided_batch['reverse'][-1].append(batch[i+1])
            elif batch[i][1] == batch[i+1][0]: # we detect a forward sequence
                detect_reverse = False
                if not detect_forward:
                    detect_forward = True # a forward sequence storage starts
                    divided_batch['forward'].append([batch[i], batch[i + 1]])
                else:   # append the next element into this sequence
                    divided_batch['forward'][-1].append(batch[i+1])
            else:
                if i == 0: # if now it is at the start of batch, then this element is not part of any sequence
                    divided_batch['single'].append(batch[i])
                else:
                    if not detect_forward and not detect_reverse: # if it isn't the tail of any sequence
                        divided_batch['single'].append(batch[i])
                detect_forward = False
                detect_reverse = False
        # if the last element is not contained in a sequence, we should add it to 'single' component
        if (not detect_forward) and (not detect_reverse):
            divided_batch['single'].append(batch[-1])

        divided_history.append(divided_batch)
        # record the sequence length
        sequence_length['forward'].extend([len(seq) for seq in divided_batch['forward']])
        sequence_length['reverse'].extend([len(seq) for seq in divided_batch['reverse']])

    if fig_path is not None:
        # draw sequence histograms where the x-axis is the length of the sequence and y-aixs the frequency
        fig, axs = plt.subplots(1,2, figsize=(8,6))
        for i, type in enumerate(['forward', 'reverse']):
            data = sequence_length[type]
            if len(data) > 0:
                axs[i].hist(data, bins=np.linspace(min(data), max(data)+1, max(data)-min(data)+2)-0.5)
            else:
                axs[i].hist([])
            axs[i].set_xlabel('Length of the sequence')
            axs[i].set_ylabel('# of appearances')
            axs[i].grid(True)
        plt.savefig(fig_path)
    return divided_history

   
 
