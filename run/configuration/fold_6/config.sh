#!/usr/bin/env bash

start_relation="confers#SPACE#resistance#SPACE#to#SPACE#antibiotic"
DATA_PATH="/Users/nicholasjoodi/Documents/ucdavis/research/HypothesisGeneration/data/k_fold/folds/fold_0/"
use_negatives=false
use_domain=true
no_negatives=('has' 'is' 'is#SPACE#involved#SPACE#in' 'upregulated#SPACE#by#SPACE#antibiotic' 'targeted#SPACE#by' 'activates' 'represses'  )
train_file="train.txt"
log_reg_calibrate=true
use_smolt_sampling=true
predict_file=train_local.txt
