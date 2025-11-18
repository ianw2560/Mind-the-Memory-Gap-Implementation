# Mind-the-Memory-Gap-Implementation

An implementation of the paper "Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference"

## GPU Profiling

### Create Conda Environment

If you wish to create the conda env from scratch, run the following:

```
conda create -n gpuprofiling python=3.11
pip install torch==2.9.0 torchvision==0.24.0 transformers==4.57.1 pandas accelerate fastparquet tiktoken sentencepiece
```

### Initial Setup

This loads the existing conda env from this repos

```
conda env create -f newton_env.yml
conda activate gpu
```



Install nvidia nsight Ubuntu
```
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture)/ /"
sudo apt install nsight-systems
```

Install nvidia nsight RHEL

```

