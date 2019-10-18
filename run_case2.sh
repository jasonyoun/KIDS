#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# do integration until wet-lab validation is needed
python3 create_kg.py --phase='phase1' --data_path=./data/data_path_file_including_Youn.txt

# analyze wet-lab validation results
python3 create_kg.py --phase='phase2'

# do rest of the integration using the validation results
python3 create_kg.py --phase='phase3'
