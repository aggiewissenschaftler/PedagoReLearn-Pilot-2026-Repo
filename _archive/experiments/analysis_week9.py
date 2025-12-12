import pandas as pd
df = pd.read_csv("trace_results/aggregation_comparison.csv")
summary = df.groupby("mode").agg(["mean","std"])
print(summary)