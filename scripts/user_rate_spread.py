"""Post-hoc descriptive statistic for protocol §10, recorded as a deviation: spread of per-user positive rates.

The frozen analysis code does not output how much users differ in positive rate, which §10 uses to explain why
global AUC, GAUC and UAUC differ. This reads the opened test sets the same way the analysis does (dedup, tab 1,
checked against the opening record), involves no model, and is registered like any other run.
Usage: python scripts/user_rate_spread.py
"""
from __future__ import annotations

import numpy as np

from receval import config, data, registry
from receval import features as F
from receval.analysis import write

MIN_ROWS = 10  # users with fewer rows have rates too noisy to describe the spread


def main() -> None:
    with registry.logged("posthoc-user-rates", {"min_rows": MIN_ROWS}) as run:
        out_dir = config.RUNS / "posthoc" / run["run_id"]
        run["output"] = str(out_dir)
        result = {}
        for name in ("standard_test", "random_test"):
            frame = F.dedup(data.load(name))
            frame = frame[frame["tab"] == 1]
            for label in ("is_click", "long_view"):
                users = frame.groupby("user_id")[label].agg(["mean", "size"])
                kept = users[users["size"] >= MIN_ROWS]["mean"]
                result[f"{name}/{label}"] = {
                    "rows": int(len(frame)), "positive_rate": float(frame[label].mean()), "users": int(len(users)),
                    f"users_with_{MIN_ROWS}_rows": int(len(kept)),
                    "user_rate_quantiles_10_25_50_75_90": [float(q) for q in np.percentile(kept, [10, 25, 50, 75, 90])],
                    "user_rate_std": float(kept.std(ddof=1))}
        print(write(result, out_dir, "user_rates"))


if __name__ == "__main__":
    main()
