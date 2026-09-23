import pandas as pd
import numpy as np

# Load the raw results
df_000 = pd.read_csv("scratch/audit_300t_all_scenarios.csv")
df_002 = pd.read_csv("scratch/full_90_battery_eps002_results.csv")
df_005 = pd.read_csv("scratch/full_90_battery_eps005_results.csv")

print("=== RAW ROW COUNT AUDIT ===")
print(f"df_000 total rows: {len(df_000)}")
print(f"df_002 total rows: {len(df_002)}")
print(f"df_005 total rows: {len(df_005)}")

# In df_000, how many benchmark vs stress?
if "category" in df_000.columns:
    print("\ndf_000 category breakdown (k20_300):")
    cat_grps = df_000.groupby("category")["k20_300"].agg(["sum", "count"])
    print(cat_grps)
    print("df_000 Total k20_300 passed:", df_000["k20_300"].sum())

print("\ndf_002 category breakdown (k20_300):")
cat_grps2 = df_002.groupby("category", sort=False)["k20_300"].agg(["sum", "count"])
print(cat_grps2)
print("df_002 Total k20_300 passed:", df_002["k20_300"].sum())

print("\ndf_005 category breakdown (k20_300):")
cat_grps5 = df_005.groupby("category", sort=False)["k20_300"].agg(["sum", "count"])
print(cat_grps5)
print("df_005 Total k20_300 passed:", df_005["k20_300"].sum())
