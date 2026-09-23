import pandas as pd
import numpy as np

df = pd.read_csv("scratch/cat6_detailed_traces.csv")

for sc_idx in range(1, 5):
    sdf = df[df["scenario_idx"] == sc_idx]
    sc_desc = sdf["desc"].iloc[0]
    seed = sdf["seed"].iloc[0]
    d_ab = sdf["d_ab"].iloc[0]
    fail_id = sdf["fail_id"].iloc[0]
    
    # Reconnection stats
    reconn_ticks = sdf[sdf["connected"] & (sdf["tick"] >= 10)]["tick"].tolist()
    first_reconn = reconn_ticks[0] if reconn_ticks else None
    
    # Drops
    drops = 0
    prev_c = False
    drop_ticks = []
    for _, row in sdf[sdf["tick"] >= 10].iterrows():
        c = row["connected"]
        if c:
            prev_c = True
        else:
            if prev_c:
                drops += 1
                drop_ticks.append(row["tick"])
                
    k20 = all(sdf[sdf["tick"].between(280, 299)]["connected"])
    c150 = sdf[sdf["tick"] == 150]["connected"].iloc[0]
    c300 = sdf[sdf["tick"] == 299]["connected"].iloc[0]
    
    print(f"=== SCENARIO {sc_idx}: {sc_desc} ===")
    print(f"  Seed={seed}, d_AB={d_ab}m, Fail Agent={fail_id}@t=10")
    print(f"  First Reconnect: t={first_reconn}")
    print(f"  Conn@150: {c150}, Conn@300: {c300}, K20 Sustained [280-300]: {k20}")
    print(f"  Drops count: {drops}, Drop ticks: {drop_ticks}")
    
    # Look at chain hops at t=50, 100, 150, 200, 220, 230, 240, 260, 280, 299
    sample_ticks = [50, 100, 150, 200, 220, 230, 240, 260, 280, 299]
    print(f"  Snapshots:")
    for st in sample_ticks:
        r = sdf[sdf["tick"] == st].iloc[0]
        print(f"    t={st:3d} | Conn={str(r['connected']):<5s} | MaxSpeed={r['max_speed']:.3f}m/s | MaxHop={r['max_chain_link']} ({r['max_chain_dist']:.2f}m)")
        print(f"           Chain: {r['chain_dists_str']}")
    print()
