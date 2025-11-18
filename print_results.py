#!/usr/bin/env python3

import pandas as pd

df = pd.read_parquet("profiling_results.parquet")

print(df)

