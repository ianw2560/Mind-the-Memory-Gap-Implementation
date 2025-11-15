import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_batchsize_vs_time(model, dtype, output_tokens, prompt_len, parquet_filename, save_dir):

    df = pd.read_parquet(parquet_filename)

    print(df)

    idx = pd.IndexSlice

    sub = df.loc[idx[model, dtype, :, output_tokens, prompt_len], :]
    batch_sizes = sub.index.get_level_values("batch")

    prefill_times = []
    decode_times = []

    rows = []
    for i in batch_sizes:
        rows.append( (model, dtype, i, output_tokens, prompt_len) )

    for i in range(len(batch_sizes)):
        prefill_times.append(df.loc[rows[i]]["prefill_time"])
        decode_times.append(df.loc[rows[i]]["decode_time"])

    x = np.arange(len(batch_sizes))

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.bar(x, prefill_times, label="Prefill Time")
    ax.bar(x, decode_times, bottom=prefill_times, label="Decode Time")

    ax.set_xlabel("Average Batch Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Batch Size vs Prefill/Decode Time (opt-1.3b, NVIDIA H100)") 

    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)

    ax.legend()
    plt.tight_layout()

    plt.savefig(f"{save_dir}/batchsize_vs_time.png", dpi=300)

def plot_throughput_vs_batchsize(model, dtype, output_tokens, prompt_len, parquet_filename, save_dir):

    df = pd.read_parquet(parquet_filename)

    model_name = model.split("/")[1]

    idx = pd.IndexSlice

    sub = df.loc[idx[model, dtype, :, output_tokens, prompt_len], :]
    batch_sizes = sub.index.get_level_values("batch")

    throughputs = []

    rows = []
    for i in batch_sizes:
        rows.append( (model, dtype, i, output_tokens, prompt_len) )

    for i in range(len(batch_sizes)):
        throughputs.append(df.loc[rows[i]]["tokens_sec"])

    x = np.arange(len(batch_sizes))

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.plot(x, throughputs, label=f"{model_name}", marker="o")

    ax.set_xlabel("Average Batch Size")
    ax.set_ylabel("Throughput (token/sec))")
    ax.set_title("Throughput vs Batchsize (NVIDIA H100)") 

    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)

    ax.legend()
    plt.tight_layout()

    plt.savefig(f"{save_dir}/throughput_vs_batchsize.png", dpi=300)


def create_plots(model, dtype, output_tokens, prompt_len, parquet_filename, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    plot_batchsize_vs_time(model, dtype, output_tokens, prompt_len, parquet_filename, save_dir)
    plot_throughput_vs_batchsize(model, dtype, output_tokens, prompt_len, parquet_filename, save_dir)


create_plots(
    model="facebook/opt-1.3b",
    dtype="fp16",
    prompt_len=128,
    output_tokens=256,
    parquet_filename="profiling_results.parquet",
    save_dir="images",
)
