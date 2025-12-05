import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from bca import calculate_b_opt, estimate_slo

def plot_batchsize_vs_time(model, dtype, output_tokens, input_tokens, parquet_filename, save_dir):

    df = pd.read_parquet(parquet_filename)

    # Parse the model name
    model_name = model.split("/")[-1]

    idx = pd.IndexSlice

    sub = df.loc[idx[model, dtype, :, output_tokens, input_tokens], :]
    batch_sizes = sub.index.get_level_values("batch")

    prefill_times = []
    decode_times = []

    rows = []
    for i in batch_sizes:
        rows.append( (model, dtype, i, output_tokens, input_tokens) )

    for i in range(len(batch_sizes)):
        prefill_times.append(df.loc[rows[i]]["prefill_time"])
        decode_times.append(df.loc[rows[i]]["decode_time"])

    x = np.arange(len(batch_sizes))

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.bar(x, prefill_times, label="Prefill Time")
    ax.bar(x, decode_times, bottom=prefill_times, label="Decode Time")

    ax.set_xlabel("Average Batch Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Batch Size vs Prefill/Decode Time ({model_name})") 

    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)

    ax.legend()
    plt.tight_layout()


    plt.savefig(f"{save_dir}/{model_name}_batchsize_vs_time.png", dpi=300)


def plot_throughput_vs_batchsize(dtype, output_tokens, input_tokens, parquet_filename, save_dir):
    df = pd.read_parquet(parquet_filename)

    # Filter on dtype, output_tokens, input_tokens
    sub_all = df.xs(
        (dtype, output_tokens, input_tokens),
        level=("dtype", "output_tokens", "input_tokens")
    )

    models = sub_all.index.get_level_values("model").unique()

    # ----------------------------------------------------------------------
    # Throughput vs Batch Size
    # ----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    for m in models:
        sub_m = sub_all.xs(m, level="model").sort_index(level="batch")
        batch_sizes = sub_m.index.get_level_values("batch")
        throughputs = sub_m["tokens_sec"]

        label = m.split("/")[-1]
        ax.plot(batch_sizes, throughputs, marker="o", label=label)

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Throughput vs Batch Size")
    ax.set_xlim(0, 600)
    ax.set_xticks(np.arange(0, 601, 100))
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/throughput_vs_batchsize.png", dpi=300)
    plt.close(fig)

    # ----------------------------------------------------------------------
    # Latency vs Batch Size (inter-token latency in ms)
    # ----------------------------------------------------------------------
    fig_lat, ax_lat = plt.subplots(figsize=(4.5, 3.5))

    for m in models:
        sub_m = sub_all.xs(m, level="model").sort_index(level="batch")
        batch_sizes = sub_m.index.get_level_values("batch")
        latencies = sub_m["inter_token_latency_ms"]

        label = m.split("/")[-1]
        ax_lat.plot(batch_sizes, latencies, marker="o", label=label)

    ax_lat.set_xlabel("Batch Size")
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_title("Latency vs Batch Size")
    ax_lat.set_xlim(0, 600)
    ax_lat.set_xticks(np.arange(0, 601, 100))
    ax_lat.grid(True)
    ax_lat.legend()
    plt.tight_layout()

    plt.savefig(f"{save_dir}/latency_vs_batchsize.png", dpi=300)
    plt.close(fig_lat)

def plot_throughput_vs_latency(model, dtype, output_tokens, input_tokens, parquet_filename, save_dir):

    df = pd.read_parquet(parquet_filename)

    # Parse the model name
    model_name = model.split("/")[-1]

    idx = pd.IndexSlice

    # Select rows for this configuration and sort by batch size
    sub = df.loc[idx[model, dtype, :, output_tokens, input_tokens], :].sort_index(level="batch")
    batch_sizes = sub.index.get_level_values("batch").to_numpy()
    

    throughputs = []
    latencies = []
    decode_times = []

    for b in batch_sizes:
        row = (model, dtype, b, output_tokens, input_tokens)
        row_data = df.loc[row]

        throughputs.append(row_data["tokens_sec"])
        latencies.append(row_data["inter_token_latency_ms"])
        decode_times.append(row_data["decode_time"])

    latencies = np.array(latencies, dtype=float)
    throughputs = np.array(throughputs, dtype=float)

    fig, ax = plt.subplots(figsize=(5, 4))

    # Throughput vs latency curve (default matplotlib color & style)
    ax.plot(latencies, throughputs, marker="o")

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title(f"Throughput vs Latency Trade-off ({model_name})")
    ax.grid(True)

    # BCA User Parameters    
    slo = estimate_slo(batch_sizes, latencies, 2.0)
    eps = 0.1

    b_opt = calculate_b_opt(batch_sizes, throughputs, latencies, slo, eps)
    b_opt = b_opt["Bopt"]

    print("B_opt:", b_opt)


    idx_opt = np.where(batch_sizes == b_opt)[0][0]
    latency_opt = latencies[idx_opt]
    ax.axvline(latency_opt, linestyle="--", label=f"B_opt: {b_opt}")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_throughput_vs_latency.png", dpi=300)
    plt.close(fig)

def plot_throughput_vs_gpu_usage(dtype, output_tokens, input_tokens,
                                 parquet_filename, gpu_mem_bytes,
                                 save_path=None):

    df = pd.read_parquet(parquet_filename)
    idx = pd.IndexSlice

    # Slice all models for given dtype / out / in
    sub_all = df.loc[idx[:, dtype, :, output_tokens, input_tokens], :].reset_index()
    sub_all = sub_all.sort_values(["model", "batch"])

    # Compute % of GPU memory used by peak_alloc
    sub_all["peak_alloc_pct_of_gpu"] = (
        100.0 * sub_all["peak_alloc_bytes"] / gpu_mem_bytes
    )

    models = sub_all["model"].unique()

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    for m in models:
        sub_m = sub_all[sub_all["model"] == m]
        x = sub_m["peak_alloc_pct_of_gpu"]
        y = sub_m["tokens_sec"]
        label = m.split("/")[-1]
        ax.plot(x, y, marker="o", label=label)

    ax.set_xlabel("Maximuim GPU Memory Usage (%)")
    ax.set_ylabel("Throughput (tokens/sec)")
    #ax.set_title(f"Throughput vs GPU Usage (dtype={dtype}, out={output_tokens}, in={input_tokens})")
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),  # x=center, y=just above the axes
        ncol=len(labels),            # spread models in a row
        frameon=False                # no box around legend (optional)
    )
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)


def print_tradeoff_table(model, dtype, output_tokens, input_tokens,
                         parquet_filename, gpu_mem_bytes=None, save_dir="images",
                         slo_multiplier=2.0, eps=0.1):
    """
    Prints & returns a DataFrame comparing percent of max throughput vs
    percent of GPU memory used by peak_alloc (in GB), and highlights B_opt.
    """
    import numpy as np
    import pandas as pd

    GB = 1024 ** 3  # GiB

    # Try to detect total GPU memory if not given
    if gpu_mem_bytes is None:
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_mem_bytes = pynvml.nvmlDeviceGetMemoryInfo(h).total
        except Exception:
            gpu_mem_bytes = np.nan  # will yield NaN % if we can't fetch it

    # Load + slice
    df = pd.read_parquet(parquet_filename)
    idx = pd.IndexSlice
    sub = df.loc[idx[model, dtype, :, output_tokens, input_tokens], :].reset_index()
    sub = sub.sort_values("batch")

    # % of max throughput
    tps_max = sub["tokens_sec"].max()
    sub["throughput_pct_of_max"] = 100.0 * sub["tokens_sec"] / tps_max

    # Peak alloc in GB (two decimals when printing)
    sub["peak_alloc_gb"] = sub["peak_alloc_bytes"] / GB

    # % of total GPU memory (peak_alloc proxy)
    if np.isfinite(gpu_mem_bytes) and gpu_mem_bytes > 0:
        sub["peak_alloc_pct_of_gpu"] = 100.0 * sub["peak_alloc_bytes"] / gpu_mem_bytes
    else:
        sub["peak_alloc_pct_of_gpu"] = np.nan

    # Compute B_opt under SLO
    batches = sub["batch"].to_numpy()
    lats    = sub["inter_token_latency_ms"].to_numpy()
    tps     = sub["tokens_sec"].to_numpy()

    slo = estimate_slo(batches, lats, slo_multiplier)
    bopt = calculate_b_opt(batches, tps, lats, slo, eps)["Bopt"]
    sub["is_Bopt"] = (sub["batch"] == bopt)

    # Pretty print a compact table
    cols = ["batch", "tokens_sec", "throughput_pct_of_max",
            "peak_alloc_gb", "peak_alloc_pct_of_gpu",
            "inter_token_latency_ms", "is_Bopt"]
    table = sub[cols].copy()

    def _fmt_pct(x): return f"{x:.2f}%" if pd.notna(x) else "NaN"
    def _fmt_gb(x): return f"{x:.2f} GB" if pd.notna(x) else "NaN"

    model_name = model.split("/")[-1]
    print(f"\n=== Throughput vs Peak-Alloc Trade-off ({model_name}) ===")
    print(table.to_string(
        index=False,
        formatters={
            "throughput_pct_of_max": _fmt_pct,
            "peak_alloc_gb": _fmt_gb,
            "peak_alloc_pct_of_gpu": _fmt_pct
        }
    ))

    # One-line summary in the paper's style (GB)
    row_opt = table.loc[table["is_Bopt"]].iloc[0]
    tps_pct = row_opt["throughput_pct_of_max"]
    mem_pct = row_opt["peak_alloc_pct_of_gpu"]
    gpu_total_gb = (gpu_mem_bytes / GB) if np.isfinite(gpu_mem_bytes) else np.nan
    print(f"\nB_opt = {int(bopt)} achieves {tps_pct:.2f}% of max throughput "
          f"while using {mem_pct:.2f}% of GPU memory (peak_alloc). "
          f"[peak_alloc={row_opt['peak_alloc_gb']:.2f} GB, "
          f"GPU total={(f'{gpu_total_gb:.2f} GB' if np.isfinite(gpu_total_gb) else 'NaN')}]")

    plot_throughput_vs_gpu_usage(
        dtype=dtype,
        output_tokens=output_tokens,
        input_tokens=input_tokens,
        parquet_filename=parquet_filename,
        gpu_mem_bytes=gpu_mem_bytes,
        save_path=f"{save_dir}/throughput_vs_gpu_usage.png"
    )

    return table


def create_plots(dtype, output_tokens, input_tokens, parquet_filename, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_parquet(parquet_filename)
    
    # Filter on dtype, output_tokens, input_tokens
    sub_all = df.xs(
        (dtype, output_tokens, input_tokens),
        level=("dtype", "output_tokens", "input_tokens")
    )

    models = sub_all.index.get_level_values("model").unique()

    for m in models:
        plot_batchsize_vs_time(m, dtype, output_tokens, input_tokens, parquet_filename, save_dir)
        plot_throughput_vs_latency(m, dtype, output_tokens, input_tokens, parquet_filename, save_dir)
        print_tradeoff_table(
            m, dtype, output_tokens, input_tokens,
            parquet_filename,   # same file you already pass around
            gpu_mem_bytes=8e10, # or pass an int if you want to fix it
            slo_multiplier=2.0,
            eps=0.1
        )


    plot_throughput_vs_batchsize(dtype, output_tokens, input_tokens, parquet_filename, save_dir)


create_plots(
    dtype="fp16",
    input_tokens=128,
    output_tokens=256,
    parquet_filename="results/final_results.parquet",
    save_dir="images",
)
