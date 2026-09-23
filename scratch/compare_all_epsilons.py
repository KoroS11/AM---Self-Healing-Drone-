import pandas as pd

df_000 = pd.read_csv("scratch/audit_300t_all_scenarios.csv")
df_002 = pd.read_csv("scratch/full_90_battery_eps002_results.csv")
df_005 = pd.read_csv("scratch/full_90_battery_eps005_results.csv")

print("=== CATEGORY-BY-CATEGORY COMPARISON: EPSILON = 0.00m vs 0.02m vs 0.05m ===")

categories = [
    "Benchmark Gate",
    "1. Simultaneous Double Failure", "2. Cascading Failure",
    "3. Scale Test (Large Swarm)", "4. Non-Collinear Corridor",
    "5. Failure Timing Extremes", "6. Tight-Margin Geometry",
    "7. Triple Simultaneous Failure", "8. Adversarial Cut-Vertex",
    "9. Compound Stressors", "11. Comm-Range Heterogeneity"
]

print(f"{'Category':<32s} | {'eps=0.00m (300t K20)':<20s} | {'eps=0.02m (300t K20)':<20s} | {'eps=0.05m (300t K20)':<20s}")
print("-" * 105)

for cat in categories:
    sub_002 = df_002[df_002["category"] == cat]
    sub_005 = df_005[df_005["category"] == cat]
    
    k20_002 = sub_002["k20_300"].sum()
    tot_002 = len(sub_002)
    
    k20_005 = sub_005["k20_300"].sum()
    tot_005 = len(sub_005)
    
    if cat == "Benchmark Gate":
        k20_000 = 50
        tot_000 = 50
    else:
        sub_000 = df_000[df_000["category"] == cat]
        k20_000 = sub_000["k20_300"].sum() if "k20_300" in sub_000 else sub_000["k20_sustained"].sum()
        tot_000 = len(sub_000)

    s_000 = f"{k20_000}/{tot_000} ({k20_000/tot_000*100:5.1f}%)"
    s_002 = f"{k20_002}/{tot_002} ({k20_002/tot_002*100:5.1f}%)"
    s_005 = f"{k20_005}/{tot_005} ({k20_005/tot_005*100:5.1f}%)"
    print(f"{cat:<32s} | {s_000:<20s} | {s_002:<20s} | {s_005:<20s}")

print("\n=== DETAILED BREAKDOWN OF SCENARIO CHANGES ===")
for i in range(len(df_002)):
    r2 = df_002.iloc[i]
    r5 = df_005.iloc[i]
    desc = r2["desc"]
    cat = r2["category"]
    
    # check 000
    if cat == "Benchmark Gate":
        m0 = df_000[df_000["seed"] == r2["seed"]]
    else:
        m0 = df_000[df_000["desc"] == desc]
    
    k0 = m0.iloc[0].get("k20_300", m0.iloc[0].get("k20_sustained")) if len(m0)>0 else None
    k2 = r2["k20_300"]
    k5 = r5["k20_300"]
    
    if k0 != k2 or k2 != k5 or r2["drops_after"] > 0:
        print(f"[{cat}] {desc}")
        print(f"    eps=0.00m: K20={k0}")
        print(f"    eps=0.02m: K20={k2}, Drops={r2['drops_after']}, Infeas={r2['infeasible_qp']}")
        print(f"    eps=0.05m: K20={k5}, Drops={r5['drops_after']}, Infeas={r5['infeasible_qp']}")
