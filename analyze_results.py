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

    model_name = model.split("/")[-1]
    plt.savefig(f"{save_dir}/{model_name}_batchsize_vs_time.png", dpi=300)


def plot_throughput_vs_batchsize(dtype, output_tokens, prompt_len, parquet_filename, save_dir):

    df = pd.read_parquet(parquet_filename)

    # Filter on dtype, output_tokens, prompt_len
    sub_all = df.xs(
        (dtype, output_tokens, prompt_len),
        level=("dtype", "output_tokens", "prompt_len")
    )

    models = sub_all.index.get_level_values("model").unique()

    fig, ax = plt.subplots(figsize=(6, 3.5))

    for m in models:
        sub_m = sub_all.xs(m, level="model").sort_index(level="batch")
        batch_sizes = sub_m.index.get_level_values("batch")
        throughputs = sub_m["tokens_sec"]

        label = m.split("/")[-1]
        ax.plot(batch_sizes, throughputs, marker="o", label=label)

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Throughput vs Batch Size (NVIDIA H100)")
    ax.set_xlim(0, 600)
    ax.set_xticks(np.arange(0, 301, 100))

    ax.legend()
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/throughput_vs_batchsize.png", dpi=300)
    plt.close(fig)


def create_plots(dtype, output_tokens, prompt_len, parquet_filename, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_parquet(parquet_filename)
    
    # Filter on dtype, output_tokens, prompt_len
    sub_all = df.xs(
        (dtype, output_tokens, prompt_len),
        level=("dtype", "output_tokens", "prompt_len")
    )

    models = sub_all.index.get_level_values("model").unique()

    for m in models:
        plot_batchsize_vs_time(m, dtype, output_tokens, prompt_len, parquet_filename, save_dir)


    plot_throughput_vs_batchsize(dtype, output_tokens, prompt_len, parquet_filename, save_dir)


create_plots(
    dtype="fp16",
    prompt_len=128,
    output_tokens=256,
    parquet_filename="batchsize_1-256_128_256_newton.parquet",
    save_dir="images",
)
