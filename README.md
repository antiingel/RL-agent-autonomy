# Quantifying reinforcement-learning agent’s autonomy, reliance on memory and internalisation of the environment

## Introduction

This repository contains the code for calculating some information-theoretic measures that characterise reinforcement-learning agents. The autonomy measures introduced by [Bertschinger et al.](https://doi.org/10.1016/j.biosystems.2007.05.018) and partial information decomposition ([BROJA2PID](https://github.com/Abzinger/BROJA_2PID) and [SxPID](https://github.com/Abzinger/SxPID)) are used. Here the measures are calculated in a practical setting for actual reinforcement-learning agents. Algorithms are introduced to calculate the measures in the process of time step approaching infinity.

The details are available in the [article](https://doi.org/10.3390/e24030401): Anti Ingel, Abdullah Makkeh, Oriol Corcoll and Raul Vicente. "Quantifying Reinforcement-Learning Agent’s Autonomy, Reliance on Memory and Internalisation of the Environment". Entropy 24.3 (2022). Please cite this article when using the code.

## Requirements

The code is written in Python and has been tested with Python 3.7. Running the code requires packages `numpy`, `scipy`, `matplotlib`, `h5py`, `discreteMarkovChain`. Code for calculating partial information decomposition is needed to run the algorithm. Please download `BROJA_2PID.py` file from [this repository](https://github.com/antiingel/BROJA_2PID) and add it to the `src` folder before trying to run the code.

## Getting started

The separate experiments for two different environments are given in folders `src/grid_environment` and `src/repeating_pattern_environment`. See the [article](https://doi.org/10.3390/e24030401) for details.

### Grid environment

Grid environment is a simple setting of Markov decision process (MDP). For grid environment the code can be ran in the order as indicated by the numbers in file names: `1_run_policy_iteration.py`, `2_calculate_autonomy.py`, `3_make_plots.py`. Note that result files are already present in folder `results`, thus one can produce plots right away by running `3_make_plots.py`.

### Repeating-pattern environment

Repeating-pattern environment is formalised as a partially observable Markov decision process (POMDP). For repeating-pattern environment the code can be ran in the order as indicated by the numbers in file names: first `1_calculate_autonomy.py` and then any other file. Note that result files are already present in the folder `results`, thus one can produce plots right away by running the corresponding scripts.

