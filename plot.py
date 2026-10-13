import csv
from collections import defaultdict

import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results/bench.csv")))
show = ["pytorch", "torch.compile", "pytorch + cuda graph", "triton, 1 kernel total", "triton + cuda graph"]

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))

by_steps = defaultdict(dict)
for r in rows:
    if r["batch"] == "1":
        by_steps[r["variant"]][int(r["steps"])] = float(r["ms"])
for name in show:
    if name in by_steps:
        steps = sorted(by_steps[name])
        left.plot(steps, [by_steps[name][s] for s in steps], marker="o", label=name)
left.set_xlabel("flow steps")
left.set_ylabel("latency per prediction (ms)")
left.set_title("Batch 1 (robot control loop)")
left.set_xticks([1, 2, 4, 8])
left.legend(fontsize=8)

at4 = {r["variant"]: float(r["ms"]) for r in rows if r["batch"] == "1" and r["steps"] == "4"}
names = [n for n in show if n in at4]
right.barh(names, [at4[n] for n in names])
right.invert_yaxis()
right.set_xlabel("ms")
right.set_title("Batch 1, 4 flow steps")

plt.tight_layout()
plt.savefig("results/latency.png", dpi=150)
print("saved results/latency.png")
