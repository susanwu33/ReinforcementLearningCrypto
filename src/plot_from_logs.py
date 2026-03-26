"""
Regenerate evaluation plots from saved test rollout CSVs (no training).

Expects files written by src.main:
  {out_dir}/{sym}_{trader_type}_pg_log.csv
  {out_dir}/{sym}_{trader_type}_dqn_log.csv

Summary-only CSVs (*_summary.csv) are not enough — you must run src.main once so the
*_pg_log.csv / *_dqn_log.csv files exist.

Run from repo root:
  python -m src.plot_from_logs --sym AAVE --out_dir results
  python -m src.plot_from_logs --auto --out_dir results   # pick up any matching logs
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from src.evaluation.plotting import generate_all_plots

# {sym}_{trader_type}_{pg|dqn}_log.csv — sym has no underscores (AAVE, SUI, …)
_LOG_NAME = re.compile(r"^([^_]+)_(.+)_(pg|dqn)_log\.csv$")


def load_saved_logs(out_dir: Path, sym: str, trader_types: list[str]) -> dict[str, pd.DataFrame]:
    all_logs: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for t in trader_types:
        for suffix, key_suffix in (("pg", "pg"), ("dqn", "dqn")):
            path = out_dir / f"{sym}_{t}_{suffix}_log.csv"
            key = f"{t}_{key_suffix}"
            if not path.is_file():
                missing.append(str(path))
                continue
            df = pd.read_csv(path)
            if "regime" in df.columns:
                df["regime"] = df["regime"].astype(str)
            all_logs[key] = df

    if missing:
        print("Warning: missing files (skipped):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)

    return all_logs


def load_discovered_logs(out_dir: Path) -> tuple[dict[str, pd.DataFrame], str]:
    """Load every *_{pg|dqn}_log.csv; sym (plot prefix) must be unique across files."""
    all_logs: dict[str, pd.DataFrame] = {}
    syms: set[str] = set()

    for path in sorted(out_dir.glob("*.csv")):
        m = _LOG_NAME.match(path.name)
        if not m:
            continue
        sym, trader, algo = m.group(1), m.group(2), m.group(3)
        syms.add(sym)
        key = f"{trader}_{algo}"
        df = pd.read_csv(path)
        if "regime" in df.columns:
            df["regime"] = df["regime"].astype(str)
        all_logs[key] = df

    if not all_logs:
        return {}, ""

    if len(syms) != 1:
        raise SystemExit(
            f"--auto found logs for multiple symbols {sorted(syms)} in {out_dir}. "
            "Use default mode with an explicit --sym and matching files, or split into separate folders."
        )

    return all_logs, syms.pop()


def list_csv_hint(out_dir: Path) -> str:
    csvs = sorted(p.name for p in out_dir.glob("*.csv"))
    if not csvs:
        return f"(no *.csv in {out_dir})"
    return "Found *.csv:\n  " + "\n  ".join(csvs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Redraw plots from saved *_pg_log.csv / *_dqn_log.csv")
    parser.add_argument("--sym", type=str, default="AAVE", help="Symbol prefix in filenames (same as src.main)")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Directory containing saved logs and where PNGs are written",
    )
    parser.add_argument(
        "--trader_types",
        nargs="+",
        default=["rational", "manipulator", "retail"],
        help="Trader types; must match filenames from the original run",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Discover all {sym}_{trader}_{pg|dqn}_log.csv in out_dir (single sym only)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        raise SystemExit(f"out_dir is not a directory: {out_dir}")

    if args.auto:
        all_logs, sym = load_discovered_logs(out_dir)
        if not all_logs:
            print(list_csv_hint(out_dir), file=sys.stderr)
            raise SystemExit(
                f"No files matching *_*_{{pg,dqn}}_log.csv under {out_dir}. "
                "Train and save logs first, e.g.  python -m src.main --sym AAVE --out_dir results"
            )
        plot_prefix = sym
    else:
        all_logs = load_saved_logs(out_dir, args.sym, args.trader_types)
        if not all_logs:
            print(list_csv_hint(out_dir), file=sys.stderr)
            raise SystemExit(
                f"No log CSVs for sym={args.sym!r} under {out_dir}. "
                f"Expected e.g. {args.sym}_rational_pg_log.csv (from src.main). "
                "Summary-only runs leave *_summary.csv but not rollout logs. "
                "Run: python -m src.main --sym AAVE --out_dir results\n"
                "Or use --auto if logs use the standard naming pattern."
            )
        plot_prefix = args.sym

    print(f"Loaded {len(all_logs)} log table(s): {', '.join(sorted(all_logs.keys()))}")

    generate_all_plots(all_logs, out_dir=str(out_dir), prefix=plot_prefix)
    print(f"Plots written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
