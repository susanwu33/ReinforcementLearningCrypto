import numpy as np
import pandas as pd


def make_final_regret_table(results):
    """
    Create final regret summary table.
    """

    rows = []

    for algorithm, result in results.items():
        final_regrets = result["regrets"][:, -1]

        mean = final_regrets.mean()
        se = final_regrets.std(ddof=1) / np.sqrt(len(final_regrets))

        ci_low = mean - 1.96 * se
        ci_high = mean + 1.96 * se

        rows.append({
            "Algorithm": algorithm,
            "Final cumulative regret": mean,
            "SE": se,
            "95% CI lower": ci_low,
            "95% CI upper": ci_high,
        })

    return pd.DataFrame(rows)