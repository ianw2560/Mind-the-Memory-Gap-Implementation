#!/bin/bash

python profile_llm.py --batch 1 --prompt_len 128 --output_tokens 256
python profile_llm.py --batch 2 --prompt_len 128 --output_tokens 256
python profile_llm.py --batch 4 --prompt_len 128 --output_tokens 256
python profile_llm.py --batch 8 --prompt_len 128 --output_tokens 256
python profile_llm.py --batch 16 --prompt_len 128 --output_tokens 256
python profile_llm.py --batch 32 --prompt_len 128 --output_tokens 256

python analyze_results.py
