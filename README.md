# Modeling the function of episodic memory in spatial learning

The code is for recreating the simulations and figures described in the paper **Modeling the function of episodic memory in spatial learning (Zeng, X., Diekmann, N., Wiskott, L., Cheng, S., 2023)**

## Description

The folder "experiments" contain the scripts for running the simulations and for plotting the figures as well as the stored data from the simulations. All other files function as supporting codes.

## Getting Started

### Installing
* Install *Python==3.7*
* Install the Cobel-RL package by following the instructions in https://github.com/sencheng/CoBeL-RL.
* Install and configure *tensorflow-gpu==2.2.0*
* Install other dependent packages: **pip install -r requirements.txt**
* Add an extra python path: **export PYTHONPATH=/your_directory/EM_Spatial_Learning**


### Executing program

* Excute **EM_Spatial_Learning/experiments/learning_compare/learning_compare.py**, **EM_Spatial_Learning/experiments/sequential_replay/sequential_replay_1.py** and **EM_Spatial_Learning/experiments/sequential_replay/sequential_replay_2.py** for the simulations
* After the simulations are done, excute **EM_Spatial_Learning/experiments/learning_compare/learning_compare_plots.py**, **EM_Spatial_Learning/experiments/sequential_replay/sequential_replay_plots.py**  and **EM_Spatial_Learning/experiments/sequential_replay/replay_analysis.py** to create the figures.


## Authors

Xiangshuai Zeng (xiangshuai.zeng@gmail.com)


