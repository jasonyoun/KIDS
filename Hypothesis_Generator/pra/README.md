# The Path Ranking Algorithm

The implementation of the PRA. This model leverages random walks to train a model that learns which paths are most important in the knowledge graph when identifying whether an edge should exist. This portion of the code is leveraging the Ni et al. implementation, along with helper scripts that I've created to handle the output of the model. The helper scripts are written in bash and python.

## Folder structure

* [data_handler](https://github.com/IBPA/Hypothesis-Generation/tree/master/pra/data_handler) - data handling

* [io_util](https://github.com/IBPA/Hypothesis-Generation/tree/master/pra/io_util) - io utils to work with the output of the PRA

* [model](https://github.com/IBPA/Hypothesis-Generation/tree/master/pra/model) - code to create model instances. This is where a user can create, train, evaluate, and predict using the PRA. A separate README explains this folder in more detail.

* [pra_imp](https://github.com/IBPA/Hypothesis-Generation/tree/master/pra/pra_imp) - code for the PRA implementation. This is the Ni et al. implementation.
