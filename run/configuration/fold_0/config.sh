#!/usr/bin/env bash

# this simply needs to be one of the relations that exists  in your test set
start_relation="confers#SPACE#resistance#SPACE#to#SPACE#antibiotic"

# The path to the directory of your data
DATA_PATH="/home/jyoun/Jason/Research/KBase/Hypothesis_Generator/data/k_fold/folds/fold_0/"

# This is deprecated
use_negatives=false

# Whether or not your entities have domains
use_domain=true

# If you were to use negatives, which ones have negatives. This is deprecated
no_negatives=('has' 'is' 'is#SPACE#involved#SPACE#in' 'upregulated#SPACE#by#SPACE#antibiotic' 'targeted#SPACE#by' 'activates' 'represses'  )

# The file name of your train data
train_file="train.txt"

# If we are calibrating the scores, what calibration model are we using. True for logistic regression, false for isotonic regression
log_reg_calibrate=true

# use smote sampling if the data in dev is unbalanced.
use_smolt_sampling=true

# The name of the file that will be predicted on.
predict_file=train_local.txt
