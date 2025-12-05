# GPU Profiling and an Implementation of a Batching Configuration Advisor for LLM Inference

An implementation of the GPU profiling and Batching Configuration Advisor (BCA) from the paper "Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference"

## Setup

Create the following conda environment to run `gpu_profiler.py`:

```
conda create -n gpuprofiler python=3.11
pip install torch==2.9.0 torchvision==0.24.0 transformers==4.57.1 pandas accelerate fastparquet tiktoken sentencepiece
conda activate gpuprofiler
```

## Run GPU Profiler

To run the GPU profiler on the UCF Newton Cluster:

```
sbatch gpu_profiler.sbatch
```

## BCA

To run the BCA, ensure GPU profiling results exist and are stored in a `.parquet` file, then run the following command:

```
python bca.py
```