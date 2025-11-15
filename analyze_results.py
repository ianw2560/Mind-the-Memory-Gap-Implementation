import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_batchsize_vs_time(model, dtype, output_tokens, prompt_len, filename, save_filename):

    df = pd.read_parquet(filename)

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

    plt.savefig(save_filename, dpi=300)



plot_batchsize_vs_time(
    model="facebook/opt-1.3b",
    dtype="fp16",
    prompt_len=128,
    output_tokens=256,
    filename="profiling_results.parquet",
    save_filename="batch_vs_time.png",
)
