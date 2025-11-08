# Mind-the-Memory-Gap-Implementation

An implementation of the paper "Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference"

## GPU Profiling

### Initial Setup

```
conda create -n gpuprofiling python=3.10
```

Install nvidia nsight Ubuntu
```
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture)/ /"
sudo apt install nsight-systems
```

Install nvidia nsight RHEL

```

