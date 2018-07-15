# Hypothesis-Generation

Inspired by Google Knowledge Vault, this project leverages deep learning and random walks to generate features for hypothesis generation. 

## Quick start:
* First, to run these experiments, you will need to install the python dependencies located in requirements.txt. Second, we used the original PRA implementation from Ni et al. This will require Java 7 or higher.
* This project is composed of three different approaches to link prediciton. To make it easier to reproduce the experiments, please navigate to the [run](https://github.com/IBPA/Hypothesis-Generation/tree/master/run) directory that contains helper shell scripts.


## Code layout

The code is separated out into a few folders. Each folder contains its own README describing what is contained.

### At a high level:

* [er_mlp](https://github.com/IBPA/Hypothesis-Generation/tree/master/er_mlp) - The implementation of the ER-MLP in TensorFlow. A fully connected feedforward artificial neural network; Also known as a latent feature model for knowledge graph completion

* [pra](https://github.com/IBPA/Hypothesis-Generation/tree/master/pra) - The implementation of the PRA. This model leverages random walks to train a model that learns which paths are most important in the knowledge graph when identifying whether an edge should exist. This portion of the code is leveraging the Ni et al. implementation, along with helper scripts that we've created to handle the output of the model. The helper scripts are written in bash and python.

  * The PRA implementation was cloned from here: https://github.com/noon99jaki


* [stacked](https://github.com/IBPA/Hypothesis-Generation/tree/master/stacked) -  An ensemble of the PRA and ER-MLP using boosted decision stumps

* [utils](https://github.com/IBPA/Hypothesis-Generation/tree/master/utils) -  a folder containing functions to handle performance metrics for all models

* [run](https://github.com/IBPA/Hypothesis-Generation/tree/master/run) -  a folder containing executable schell scripts to run experiments

