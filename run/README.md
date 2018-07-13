# run

This directory contains shell scripts to create, train, and evalue models

## run_er_mlp.sh
To create, train, and evaluate the ER-MLP over a particular experiment, run:

```
./run_er_mlp.sh dir_name
```
where dir_name represents the folder containing the er_mlp configurations. This folder is stored in the configurations subdirectory. For example, to run er_mlp over the first fold of the experiments for this project:
```
./run_er_mlp.sh fold_0
```
Notice the fold_0 directory in the configurations directory.The configuration file in this case is the er_mlp.ini file. When this script runs, the model will be created, trained, and evaluated over the fold 0 data set for this project. The model and its associated files will be automatically created in [root]/er_mlp/model/model_instance/fold_0. Note: before running this script, the DATA_PATH property in the er_mlp.ini file will need to be updated to where the dataset is stored on your machine.
