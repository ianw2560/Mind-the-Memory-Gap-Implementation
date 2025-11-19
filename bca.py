#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd

def calculate_b_opt(batch_sizes, T_values, L_values, SLO, eps=0.1, tol=1e-3):
    B = np.array(batch_sizes) # Batch size
    T = np.array(T_values) # Throughput
    L = np.array(L_values) # Latency

    idx1 = np.where(B == 1)[0]
    if idx1.size > 0:
        T1 = T[idx1[0]]
    else:
        T1 = T[0]

    eff_ratio = T / (B * T1 + 1e-12)
    feasible_mask = (L <= SLO) & (eff_ratio > eps)

    # Satisfy latency and throughput conditions
    feasible_B = B[feasible_mask]
    feasible_T = T[feasible_mask]
    feasible_L = L[feasible_mask]

    if feasible_B.size == 0:
        return {
            'Bopt': None,
            'T': None,
            'L': None,
            'feasible_list': []
        }

    maxT = feasible_T.max()
    near_mask = (feasible_T >= maxT * (1 - tol))
    near_B = feasible_B[near_mask]
    near_T = feasible_T[near_mask]
    near_L = feasible_L[near_mask]

    pick_idx = np.argmin(near_B)
    Bopt = int(near_B[pick_idx])
    Topt = float(near_T[pick_idx])
    Lopt = float(near_L[pick_idx])

    feasible_list = sorted(
        [{'B': int(b), 'T': float(t), 'L': float(l)} 
         for b,t,l in zip(feasible_B, feasible_T, feasible_L)],
        key=lambda x: x['B']
    )

    return {
        'Bopt': Bopt,
        'T': Topt,
        'L': Lopt,
        'feasible_list': feasible_list
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b", help="Model to use")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16","bf16","fp32","int8","int4"])
    parser.add_argument("--input_tokens", type=int, default=128, help="Number of input tokens")
    parser.add_argument("--output_tokens", type=int, default=256, help="Number of output tokens to generate")
    parser.add_argument("--input_filename", type=str, default="profiling_results.parquet")
    parser.add_argument("--all", action='store_true', help="Iterate overall models from 1 to batch.")
    args = parser.parse_args()

    # BCA User Parameters    
    SLO = 10.0 #25.0
    eps = 0.1

    # Read in GPU profiling data
    df = pd.read_parquet(args.input_filename)
    
    # Filter on dtype, output_tokens, input_tokens
    model_df = df.xs(
        (args.model, args.dtype, args.output_tokens, args.input_tokens),
        level=("model", "dtype", "output_tokens", "input_tokens")
    )

    batch_sizes = model_df.index.to_numpy()
    latencies = model_df["inter_token_latency_ms"].to_numpy()
    decode_times = model_df["decode_time"].to_numpy()

    print(batch_sizes)
    print(latencies)
    print(decode_times)

    result = calculate_b_opt(batch_sizes, decode_times, latencies, SLO, eps)

    print("=== RESULTS ===")
    print("Bopt:", result['Bopt'])
    print("T:", result['T'])
    print("L:", result['L'])
    print("Feasable Values:", result['feasible_list'])
