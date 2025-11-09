#!/bin/bash

python profiler.py --batch 1 --prompt_len 128 --output_tokens 256
python profiler.py --batch 2 --prompt_len 128 --output_tokens 256
python profiler.py --batch 4 --prompt_len 128 --output_tokens 256
python profiler.py --batch 8 --prompt_len 128 --output_tokens 256
python profiler.py --batch 16 --prompt_len 128 --output_tokens 256
python profiler.py --batch 32 --prompt_len 128 --output_tokens 256

python analyze_results.py
