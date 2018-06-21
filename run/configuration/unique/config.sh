#!/usr/bin/env bash

start_relation="confers#SPACE#resistance#SPACE#to#SPACE#antibiotic"
DATA_PATH="/Users/nicholasjoodi/Documents/ucdavis/research/HypothesisGeneration/data/kb_head_just_negatives_05_21_1/"
use_negatives=false
use_domain=true
no_negatives=('has' 'is' 'is#SPACE#involved#SPACE#in' 'upregulated#SPACE#by#SPACE#antibiotic' 'targeted#SPACE#by' 'activates' 'represses'  )
train_file="train.txt"
log_reg_calibrate=true
use_smolt_sampling=true
predict_file=train_local.txt
