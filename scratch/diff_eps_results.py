import pandas as pd

df_old = pd.read_csv("scratch/audit_300t_all_scenarios.csv")
df_new = pd.read_csv("scratch/full_90_battery_eps005_results.csv")

print("=== COMPARISON OF EVERY SCENARIO (EPSILON=0.00m vs EPSILON=0.05m) ===")

# Match by desc or category/seed
for i in range(len(df_new)):
    r_new = df_new.iloc[i]
    # find matching in df_old
    m = df_old[df_old["desc"] == r_new["desc"]]
    if len(m) == 0:
        m = df_old[df_old["seed"] == r_new["seed"]]
    
    if len(m) > 0:
        r_old = m.iloc[0]
        k_old = r_old.get("k20_300", r_old.get("k20_sustained"))
        k_new = r_new["k20_300"]
        c_old = r_old.get("snap_300", r_old.get("reconnected"))
        c_new = r_new["snap_300"]
        drops_old = r_old.get("drops_after", r_old.get("transient_drops"))
        drops_new = r_new["drops_after"]
        
        if k_old != k_new or c_old != c_new or drops_old != drops_new:
            print(f"DIFFERENCE in '{r_new['desc']}' (Cat: {r_new['category']}):")
            print(f"  eps=0.00m: K20_300={k_old}, Snap300={c_old}, Drops={drops_old}")
            print(f"  eps=0.05m: K20_300={k_new}, Snap300={c_new}, Drops={drops_new}, Infeas={r_new['infeasible_qp']}")
