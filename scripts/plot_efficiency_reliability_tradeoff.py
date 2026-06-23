from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "tables" / "efficiency_reliability_main.csv"
OUT_PDF = ROOT / "paper" / "figures" / "efficiency_reliability_tradeoff.pdf"
OUT_PNG = ROOT / "paper" / "figures" / "efficiency_reliability_tradeoff.png"


def first_float(text: str) -> float:
    return float(str(text).split("+/-")[0].strip().rstrip("%"))

rows = []
with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        row["acc_value"] = first_float(row["accuracy"])
        row["ece_value"] = first_float(row["ece"])
        row["speed_value"] = first_float(row["wall_clock_speedup_proxy"])
        rows.append(row)

baselines = {(r["model"], r["dataset"]): r for r in rows if r["pruner"] == "none"}
points = []
for row in rows:
    if row["pruner"] == "none":
        continue
    base = baselines.get((row["model"], row["dataset"]))
    if base is None:
        continue
    ece_delta = row["ece_value"] - base["ece_value"]
    acc_drop = max(0.0, base["acc_value"] - row["acc_value"])
    label = f"{row['model'].replace('_4bit','')}\n{row['dataset']} {row['pruner']} r={row['retention']}"
    points.append((row["speed_value"], ece_delta, acc_drop, label))

fig, ax = plt.subplots(figsize=(7.2, 4.5))
for speed, ece_delta, acc_drop, label in points:
    size = 80 + 6000 * acc_drop
    color = "#0072B2" if ece_delta <= 0 else "#D55E00"
    ax.scatter(speed, ece_delta, s=size, color=color, alpha=0.8, edgecolor="black", linewidth=0.6)
    ax.annotate(label, (speed, ece_delta), xytext=(5, 5), textcoords="offset points", fontsize=7)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
ax.set_xlabel("Wall-clock speedup proxy vs. matched unpruned run (%)")
ax.set_ylabel("ECE change vs. matched unpruned run")
ax.set_title("Efficiency-reliability trade-off under visual token pruning")
ax.grid(True, color="#DDDDDD", linewidth=0.6)
ax.text(0.01, -0.22, "Positive x means faster in runner-level wall-clock logs; positive y means worse ECE.", transform=ax.transAxes, fontsize=8)
fig.tight_layout()
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=200)
