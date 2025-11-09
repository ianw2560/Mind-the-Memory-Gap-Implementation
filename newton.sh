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
ncu --target-processes all \
--section MemoryWorkloadAnalysis \
--section SpeedOfLight_RooflineChart \
--metrics dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.pct_of_peak_sustained_active \
--nvtx \
--nvtx-include "decode" \
--csv -o ncu_decode_b1 \
python profile_llm.py --batch 1 --prompt_len 128 --output_tokens 256