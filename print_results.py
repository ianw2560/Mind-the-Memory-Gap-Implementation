#!/usr/bin/env python3

import pandas as pd

df = pd.read_parquet("profiling_results.parquet")
# df = df.sort_index()
# print(df.index.nlevels)   # probably 5
# print(df.index.names) 


# If it has 5 levels and the last one is unnamed, name it.
# if df.index.nlevels == 5 and df.index.names[-1] is None:
#     df.index = df.index.set_names("input_tokens", level=-1)


# df.to_parquet("final_results.parquet")

print(df)

