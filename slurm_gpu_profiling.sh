#!/bin/bash -l

#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH -o batchsize_%j.out

time python profiler.py --all
