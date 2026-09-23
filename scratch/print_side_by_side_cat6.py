import pandas as pd

df = pd.read_csv("scratch/cat6_exact_traces.csv")

# Filter ticks 200..299
df_sub = df[df["tick"].between(200, 299)]

# Pivot tables for side-by-side comparison
conn_piv = df_sub.pivot(index="tick", columns="scenario_idx", values="connected")
link_piv = df_sub.pivot(index="tick", columns="scenario_idx", values="max_chain_link")
dist_piv = df_sub.pivot(index="tick", columns="scenario_idx", values="max_chain_d")
slack_piv = df_sub.pivot(index="tick", columns="scenario_idx", values="min_chain_slack")
speed_piv = df_sub.pivot(index="tick", columns="scenario_idx", values="max_speed")

print("=== TICKS 200-300 SIDE-BY-SIDE SUMMARY ===")
print("Tick | Sc1 (185m) Conn/Link/Slack/Spd | Sc2 (189m) Conn/Link/Slack/Spd | Sc3 (192m) Conn/Link/Slack/Spd | Sc4 (195m) Conn/Link/Slack/Spd")
for t in range(200, 300):
    s1 = f"{str(conn_piv.loc[t, 1])[0]}|{link_piv.loc[t, 1]}|{slack_piv.loc[t, 1]:+5.2f}m|{speed_piv.loc[t, 1]:4.2f}"
    s2 = f"{str(conn_piv.loc[t, 2])[0]}|{link_piv.loc[t, 2]}|{slack_piv.loc[t, 2]:+5.2f}m|{speed_piv.loc[t, 2]:4.2f}"
    s3 = f"{str(conn_piv.loc[t, 3])[0]}|{link_piv.loc[t, 3]}|{slack_piv.loc[t, 3]:+5.2f}m|{speed_piv.loc[t, 3]:4.2f}"
    s4 = f"{str(conn_piv.loc[t, 4])[0]}|{link_piv.loc[t, 4]}|{slack_piv.loc[t, 4]:+5.2f}m|{speed_piv.loc[t, 4]:4.2f}"
    print(f"{t:4d} | {s1:<30s} | {s2:<30s} | {s3:<30s} | {s4:<30s}")
