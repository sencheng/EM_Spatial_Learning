# The functional role of episodic memory in spatial learning

The code is for recreating the simulations and figures described in the paper **The functional role of episodic memory in spatial learning (Zeng, X., Wiskott, L., Cheng, S., 2022)**

## Description

The folder "Experiments" contain the scripts for running the simulations, and the folder "data" contains the scripts for plotting the figures as well as the stored data from the simulations. All other files function as supporting codes.

## Getting Started

### Installing

* Install *Python==3.6*
* Install and configure *tensorflow-gpu==2.2.0*
* Install other dependent packages: **pip install -r requirements.txt**
* Add an extra python path: **export PYTHONPATH=/your_directory/EM_Spatial_Learning**

### Executing program

* Excute **EM_Spatial_Learning/experiments/spatial_learning_individual.py** and **EM_Spatial_Learning/experiments/spatial_learning_hybird.py** for the simulations
* After the simulations are done, excute **EM_Spatial_Learning/data/spatial_plots_individual.py** and **EM_Spatial_Learning/data/spatial_plots_hybrid.py** to create the figures.


## Authors

Xiangshuai Zeng (xiangshuai.zeng@gmail.com)


