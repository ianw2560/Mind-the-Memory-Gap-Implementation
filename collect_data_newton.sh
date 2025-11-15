#!/bin/bash -l

#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH -o profiler_output.out

# The anaconda module provides Python
#module load anaconda/anaconda-2023.09

# Activate one of the pre-made ARCC environments (or
# your own environment) on top of the base environment.

# Run a Python script
python profiler.py --batch 1 --prompt_len 128 --output_tokens 256
python profiler.py --batch 2 --prompt_len 128 --output_tokens 256
python profiler.py --batch 4 --prompt_len 128 --output_tokens 256
python profiler.py --batch 8 --prompt_len 128 --output_tokens 256
python profiler.py --batch 16 --prompt_len 128 --output_tokens 256
python profiler.py --batch 32 --prompt_len 128 --output_tokens 256
python profiler.py --batch 64 --prompt_len 128 --output_tokens 256
python profiler.py --batch 128 --prompt_len 128 --output_tokens 256
python profiler.py --batch 256 --prompt_len 128 --output_tokens 256
python profiler.py --batch 512 --prompt_len 128 --output_tokens 256

