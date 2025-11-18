#!/bin/bash -l

#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH -o profiler_output.out


echo $HUGGINGFACE_KEY

# OPT-2.7B
python profiler.py --model facebook/opt-2.7b --batch 1 --prompt_len 128 --output_tokens 256
python profiler.py --model facebook/opt-2.7b --batch 2 --prompt_len 128 --output_tokens 256
python profiler.py --model facebook/opt-2.7b --batch 4 --prompt_len 128 --output_tokens 256
python profiler.py --model facebook/opt-2.7b --batch 8 --prompt_len 128 --output_tokens 256
python profiler.py --model facebook/opt-2.7b --batch 16 --prompt_len 128 --output_tokens 256
python profiler.py --model facebook/opt-2.7b --batch 32 --prompt_len 128 --output_tokens 256
python profiler.py --model facebook/opt-2.7b --batch 64 --prompt_len 128 --output_tokens 256
python profiler.py --model facebook/opt-2.7b --batch 128 --prompt_len 128 --output_tokens 256
python profiler.py --model facebook/opt-2.7b --batch 256 --prompt_len 128 --output_tokens 256
python profiler.py --model facebook/opt-2.7b --batch 512 --prompt_len 128 --output_tokens 256

# # Llama 2.7B
# python profiler.py --model meta-llama/Llama-2-7b --batch 1 --prompt_len 128 --output_tokens 256
# python profiler.py --model meta-llama/Llama-2-7b --batch 2 --prompt_len 128 --output_tokens 256
# python profiler.py --model meta-llama/Llama-2-7b --batch 4 --prompt_len 128 --output_tokens 256
# python profiler.py --model meta-llama/Llama-2-7b --batch 8 --prompt_len 128 --output_tokens 256
# python profiler.py --model meta-llama/Llama-2-7b --batch 16 --prompt_len 128 --output_tokens 256
# python profiler.py --model meta-llama/Llama-2-7b --batch 32 --prompt_len 128 --output_tokens 256
# python profiler.py --model meta-llama/Llama-2-7b --batch 64 --prompt_len 128 --output_tokens 256
# python profiler.py --model meta-llama/Llama-2-7b --batch 128 --prompt_len 128 --output_tokens 256
# python profiler.py --model meta-llama/Llama-2-7b --batch 256 --prompt_len 128 --output_tokens 256
# python profiler.py --model meta-llama/Llama-2-7b --batch 512 --prompt_len 128 --output_tokens 256

# # Llama 2.13B
# python profiler.py --model meta-llama/Llama-2-13b --batch 1 --prompt_len 128 --output_tokens 256
# python profiler.py --model meta-llama/Llama-2-13b --batch 2 --prompt_len 128 --output_tokens 256
# python profiler.py --model meta-llama/Llama-2-13b --batch 4 --prompt_len 128 --output_tokens 256
# python profiler.py --model meta-llama/Llama-2-13b --batch 8 --prompt_len 128 --output_tokens 256
# python profiler.py --model meta-llama/Llama-2-13b --batch 16 --prompt_len 128 --output_tokens 256
# python profiler.py --model meta-llama/Llama-2-13b --batch 32 --prompt_len 128 --output_tokens 256
# python profiler.py --model meta-llama/Llama-2-13b --batch 64 --prompt_len 128 --output_tokens 256
# python profiler.py --model meta-llama/Llama-2-13b --batch 128 --prompt_len 128 --output_tokens 256
# python profiler.py --model meta-llama/Llama-2-13b --batch 256 --prompt_len 128 --output_tokens 256
# python profiler.py --model meta-llama/Llama-2-13b --batch 256 --prompt_len 128 --output_tokens 256

# # OPT-1.3B
# python profiler.py --model facebook/opt-1.3b --batch 1 --prompt_len 128 --output_tokens 256
# python profiler.py --model facebook/opt-1.3b --batch 2 --prompt_len 128 --output_tokens 256
# python profiler.py --model facebook/opt-1.3b --batch 4 --prompt_len 128 --output_tokens 256
# python profiler.py --model facebook/opt-1.3b --batch 8 --prompt_len 128 --output_tokens 256
# python profiler.py --model facebook/opt-1.3b --batch 16 --prompt_len 128 --output_tokens 256
# python profiler.py --model facebook/opt-1.3b --batch 32 --prompt_len 128 --output_tokens 256
# python profiler.py --model facebook/opt-1.3b --batch 64 --prompt_len 128 --output_tokens 256
# python profiler.py --model facebook/opt-1.3b --batch 128 --prompt_len 128 --output_tokens 256
# python profiler.py --model facebook/opt-1.3b --batch 256 --prompt_len 128 --output_tokens 256
