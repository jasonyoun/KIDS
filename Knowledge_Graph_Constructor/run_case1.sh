#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# do integration until validation is needed
python3 create_kg.py --phase='phase1'

# use validation results to generate final knowledge graph
python3 create_kg.py --phase='phase2'
