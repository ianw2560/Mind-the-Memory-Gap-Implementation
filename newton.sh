#!/bin/bash

#SBATCH --gres=gpu::1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=1:00:00

# The anaconda module provides Python
module load anaconda/anaconda-2023.09

# Activate one of the pre-made ARCC environments (or
# your own environment) on top of the base environment.
# conda activate base

# Run a Python script
python3 profile_llm.py