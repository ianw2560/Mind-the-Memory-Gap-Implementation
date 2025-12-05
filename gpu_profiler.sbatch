#!/bin/bash -l

#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o gpu_profiler_%j.out

time python gpu_profiler.py --all
