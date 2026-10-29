"""File counts of the rerun store, and bitwise checks of its single chains against the original store."""
import sys
from pathlib import Path

import numpy as np

NEW = Path(sys.argv[1] if len(sys.argv) > 1 else "/scratch/e1377027/opt/variants/ae_baseline")
OLD = Path("/scratch/e0950116/bart_store")
TAGS = ["fixed100_Concrete", "fixed100_SeoulBike", "fixed100_Abalone", "fixed100_Airfoil",
        "fixed100_CalHousing_subsample5000", "fixed100_CCPP"]


def load(path):
    with open(path) as fh:
        header = fh.readline()
    return np.loadtxt(path, delimiter=",", comments="#"), header.strip()


for tag in TAGS:
    counts = {d.name: len(list(d.glob("*.csv"))) for d in sorted((NEW / tag).iterdir())
              if d.is_dir() and d.name != "_tasks"}
    print(tag, counts, flush=True)
    for method in ("default", "mtmh"):
        verdict = []
        for folder in ("preds", "sigmas", "rmses", "vars_histogram"):
            same, missing = 0, 0
            for run in range(5):
                name = (f"{tag}__run{run:03d}__{method}__preds.csv" if folder == "preds"
                        else f"{tag}__run{run:03d}__{method}__{folder}.csv")
                new, old = NEW / tag / folder / name, OLD / tag / folder / name
                if not new.exists():
                    missing += 1
                    continue
                if not old.exists():
                    missing += 1
                    continue
                (a, ha), (b, hb) = load(new), load(old)
                same += int(ha == hb and np.array_equal(a, b))
            verdict.append(f"{folder} {same}/5 identical" + (f" ({missing} missing)" if missing else ""))
        print(f"  {method}: " + "; ".join(verdict), flush=True)
