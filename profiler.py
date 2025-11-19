#!/usr/bin/env python3

import os, time, csv, argparse, math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.cuda.nvtx as nvtx
import pandas as pd
from huggingface_hub import login

try:
    import pynvml
    pynvml.nvmlInit()
    NVML = True
    NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    NVML = False

def current_time_ms():
    return time.time() * 1000.0

def make_prompt(tokenizer, input_tokens):
    
    # Create a test prompt
    prompt = "It is an ancient Mariner, And he stoppeth one of three. 'By thy long grey beard and glittering eye, Now wherefore stopp'st thou me?"

    s = (prompt * ((input_tokens // len(tokenizer.encode(prompt))) + 2))
    tokens = tokenizer.encode(s)

    # Pad/trim to ~input_tokens tokens
    if len(tokens) < input_tokens:
        tokens = tokens + [tokenizer.eos_token_id] * (input_tokens - len(tokens))
    else:
        tokens = tokens[:input_tokens]

    return tokenizer.decode(tokens, skip_special_tokens=False)

def bytes_to_mb(x):
    return f"{x/1024/1024:.1f} MB"

def save_results(result, filename="profiling_results.parquet"):

    # Create a key to get set of profiling results
    new_row = pd.DataFrame([result]).set_index(["model", "dtype", "batch", "output_tokens", "input_tokens"])

    # Check if results already exist
    if os.path.exists(filename):

        # Load existing results
        df = pd.read_parquet(filename)

        # Concatenate and drop duplicate indices, keeping the latest
        df = pd.concat([df, new_row])
        df = df[~df.index.duplicated(keep="last")]
    else:
        df = new_row

    df.to_parquet(filename)

def profile_gpu(model_name, dtype_str, batch, input_tokens, output_tokens, output_filename):

    # Set cuda as the default device
    device = "cuda"

    # Set dtype values
    if dtype_str == "fp16":
        dtype = torch.float16
    elif dtype_str == "bf16":
        dtype = torch.bfloat16
    elif dtype_str == "fp32":
        dtype = torch.float32
    else:
        dtype = None

    print(f"[INFO] Loading model={model_name} dtype={dtype} device={device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": device}
    if dtype is not None:
        load_kwargs["dtype"] = dtype
    if dtype in ("int8","int4"):
        # Quantized load
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=(dtype=="int8"),
                                     load_in_4bit=(dtype=="int4"))
        load_kwargs["quantization_config"] = bnb_cfg

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    # Build batch of prompts of desired token length
    prompt_text = make_prompt(tokenizer, input_tokens)
    batch_prompts = [prompt_text] * batch
    enc = tokenizer(batch_prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    # Warmup (stabilize kernels & allocator)
    with torch.no_grad():
        _ = model.generate(input_ids=input_ids[:1], attention_mask=attn_mask[:1],
                           max_new_tokens=4, do_sample=False)
        torch.cuda.synchronize()

    # Memory reset
    torch.cuda.reset_peak_memory_stats()
    if NVML:
        mem0 = pynvml.nvmlDeviceGetMemoryInfo(NVML_HANDLE).used
    else:
        mem0 = None

    #-------------------------------------------------------------------------------
    # Prefill Stage
    #-------------------------------------------------------------------------------
    nvtx.range_push("prefill")
    torch.cuda.synchronize()
    t0 = current_time_ms()

    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attn_mask)

    torch.cuda.synchronize()
    t1 = current_time_ms()
    nvtx.range_pop()

    prefill_time = t1 - t0

    #-------------------------------------------------------------------------------
    # Decode Stage
    #-------------------------------------------------------------------------------
    gen_kwargs = dict(max_new_tokens=output_tokens, do_sample=False)
    nvtx.range_push("decode")
    torch.cuda.synchronize()
    t2 = current_time_ms()

    with torch.no_grad():
        _ = model.generate(input_ids=input_ids, attention_mask=attn_mask, **gen_kwargs)

    torch.cuda.synchronize()
    t3 = current_time_ms()
    nvtx.range_pop()

    decode_time = t3 - t2

    peak_alloc = torch.cuda.max_memory_allocated()
    if NVML:
        mem1 = pynvml.nvmlDeviceGetMemoryInfo(NVML_HANDLE).used
        nvml_delta = mem1 - mem0
    else:
        mem1 = None
        nvml_delta = None

    #-------------------------------------------------------------------------------
    # Calculate metrics
    #-------------------------------------------------------------------------------
    tokens = (input_tokens + output_tokens) * batch
    tps = (tokens) / (decode_time / 1000.0)

    # Inter-token latency (ITL): average time between two generated tokens
    # for a single request (ms), using only the decode phase.
    inter_token_latency_ms  = decode_time / output_tokens

    print("=== RESULTS ===")
    print(f"BatchSize={batch} PromptLength={input_tokens} OutputTokens={output_tokens} DType={dtype}")
    print(f"Prefill: {prefill_time:.1f} ms")
    print(f"Decode: {decode_time:.1f} ms")
    print(f"Decode throughput: {tps:.2f} tokens/sec")
    print(f"Inter-token latency: {inter_token_latency_ms:.2f} ms")
    print(f"Peak CUDA alloc: {bytes_to_mb(peak_alloc)}")
    if nvml_delta is not None:
        print(f"NVML used delta: {bytes_to_mb(nvml_delta)} (approx overall increase)")
    print("===============")

    result = {
        "model": model_name,
        "dtype": dtype_str,
        "batch": batch,
        "output_tokens": output_tokens,
        "input_tokens": input_tokens,
        "prefill_time": round(prefill_time, 3),
        "decode_time": round(decode_time, 3),
        "tokens_sec": round(tps, 3),
        "inter_token_latency_ms": round(inter_token_latency_ms, 3),
        "peak_alloc_bytes": int(peak_alloc),
        "nvml_used_delta_bytes": int(nvml_delta or 0)
    }

    # Add results to existing DataFrame in parquet file
    save_results(result, output_filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b", help="Model to use")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16","bf16","fp32","int8","int4"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--input_tokens", type=int, default=1024, help="Number of input tokens")
    parser.add_argument("--output_tokens", type=int, default=256, help="Number of output tokens to generate")
    parser.add_argument("--output_filename", type=str, default="profiling_results.parquet")
    parser.add_argument("--all", action='store_true', help="Iterate overall models from 1 to batch.")
    args = parser.parse_args()

    torch.manual_seed(42)

    # Login into HuggingFace for Llama models
    login(token=os.getenv("HUGGINGFACE_KEY"))

    # Check if cuda is available
    if not torch.cuda.is_available():
        print("CUDA is required for this project.")
        exit(1)

    if not args.all:
        profile_gpu(args.model, args.dtype, args.batch, args.input_tokens, args.output_tokens, args.output_filename)
    else:
        models = ["facebook/opt-1.3b", "facebook/opt-2.7b", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-7b-hf"]
        batch_sizes = [1, 2, 4, 8]
        input_tokens = 128
        output_tokens = 256

        for model in models:
            for batch in batch_sizes:
                profile_gpu(model, args.dtype, batch, input_tokens, output_tokens, args.output_filename)


if __name__ == "__main__":
    main()
