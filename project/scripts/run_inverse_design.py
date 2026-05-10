"""Driver for the inverse design agent.

Usage:
    python scripts/run_inverse_design.py --config configs/inverse_design.yaml \
        [--target-bg 1.0] [--n-iter 5] [--n-per-iter 5] [--seed 0] [--out-dir RESULTS]

If `--target-bg` is given the script runs ONE target; otherwise it sweeps all
targets listed in the config. Outputs per target are written to
``out_dir/target_{target_bg:.2f}/`` and contain:
- results.json     (full record dump, list-of-dicts)
- best_per_iter.csv (best |error| per iteration, for convergence plot)
- top_k.csv        (top-K candidates by |error| across all iters)

The script must be launched from the project root so that the relative
``cif_path`` columns in the parquet resolve correctly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset
from src.inverse_design.agent import (
    CandidateRecord,
    InverseDesignAgent,
    records_to_dataframe,
)
from src.inverse_design.oracle import EnsembleForwardOracle
from src.inverse_design.proposer import QwenProposer

logger = logging.getLogger("run_inverse_design")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/inverse_design.yaml",
                   help="Path to inverse_design.yaml (relative to project root).")
    p.add_argument("--target-bg", type=float, default=None,
                   help="Run a single target; overrides config.targets.")
    p.add_argument("--n-iter", type=int, default=None)
    p.add_argument("--n-per-iter", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", default=None,
                   help="Output root (defaults to config.output_dir).")
    p.add_argument("--smoke", action="store_true",
                   help="Mini run: 1 target, 1 iter, 2 candidates.")
    p.add_argument("--use-fallback-llm", action="store_true",
                   help="Use the text-only Qwen2.5-3B-Instruct (no VL).")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip LLM, use random fallback for every iter (for smoke debug).")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def derive_vocab(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    a_vocab = sorted(df["a_site"].dropna().unique().tolist())
    b_vocab = sorted(df["b_site"].dropna().unique().tolist())
    return a_vocab, b_vocab


def write_outputs(out_dir: Path, target_bg: float,
                  records: Sequence[CandidateRecord]):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = records_to_dataframe(records)

    # Full results
    with open(out_dir / "results.json", "w") as fh:
        json.dump(
            {"target_bg": target_bg,
             "records": [asdict(r) for r in records]},
            fh, indent=2, default=str,
        )
    df.to_csv(out_dir / "all_candidates.csv", index=False)

    # Best |error| per iter
    best_per_iter = (
        df.loc[df.groupby("iter")["error_to_target"].idxmin()]
        [["iter", "A", "B", "base_mid", "pred_bg_mean", "pred_bg_std",
          "error_to_target"]]
        .sort_values("iter")
        .reset_index(drop=True)
    )
    best_per_iter.to_csv(out_dir / "best_per_iter.csv", index=False)

    # Top-K (default 10) by |error|, prefer high-confidence and charge-balanced
    top_k = df.sort_values(
        ["error_to_target", "low_confidence", "charge_warning"]
    ).head(10).reset_index(drop=True)
    top_k.to_csv(out_dir / "top_k.csv", index=False)

    logger.info("Wrote %d records to %s (best |err|=%.3f)", len(df), out_dir,
                df["error_to_target"].min())


def run_one_target(target_bg: float, cfg: dict, args: argparse.Namespace,
                   df: pd.DataFrame, oracle: EnsembleForwardOracle,
                   proposer: QwenProposer, a_vocab: List[str],
                   b_vocab: List[str], out_root: Path):
    n_iter = args.n_iter if args.n_iter is not None else cfg.get("n_iter", 5)
    n_per_iter = args.n_per_iter if args.n_per_iter is not None else cfg.get("n_per_iter", 5)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    if args.smoke:
        n_iter, n_per_iter = 1, 2

    workdir = out_root / f"target_{target_bg:.2f}"
    workdir.mkdir(parents=True, exist_ok=True)

    logger.info("=== target=%.2f eV, n_iter=%d, n_per_iter=%d, seed=%d ===",
                target_bg, n_iter, n_per_iter, seed)
    agent = InverseDesignAgent(
        oracle=oracle, proposer=proposer, base_df=df,
        target_bg=target_bg, a_vocab=a_vocab, b_vocab=b_vocab,
        n_iter=n_iter, n_per_iter=n_per_iter, workdir=workdir, seed=seed,
        std_threshold=cfg.get("std_threshold", 0.5),
    )
    records = agent.run()
    write_outputs(workdir, target_bg, records)
    return records


class _RandomOnlyProposer:
    """Drop-in replacement that always falls back to random sampling.

    Useful for the smoke test where loading Qwen would be expensive and
    unrelated to the bug we're hunting.
    """

    def __init__(self, seed: int = 0):
        import random
        self._rng = random.Random(seed)

    def propose(self, target_bg, history, a_vocab, b_vocab, base_set, n_proposals):
        from src.inverse_design.proposer import random_fallback_propose
        return random_fallback_propose(
            n=n_proposals, a_vocab=a_vocab, b_vocab=b_vocab,
            base_set=base_set, rng=self._rng,
        )


def main():
    args = parse_args()
    cfg_path = (PROJECT_ROOT / args.config).resolve()
    cfg = load_config(cfg_path)

    # Resolve all paths relative to project root
    os.chdir(PROJECT_ROOT)

    # Targets
    if args.target_bg is not None:
        targets = [args.target_bg]
    else:
        targets = list(cfg.get("targets", [1.0, 2.0, 3.0]))
    if args.smoke:
        targets = targets[:1]

    out_root = Path(args.out_dir or cfg.get("output_dir", "results/inverse_design"))
    out_root.mkdir(parents=True, exist_ok=True)

    # Data
    df = load_dataset(cfg["dataset_path"], text_ok_only=True)
    a_vocab, b_vocab = derive_vocab(df)
    logger.info("Loaded %d base materials, |A|=%d, |B|=%d",
                len(df), len(a_vocab), len(b_vocab))

    # Oracle
    logger.info("Loading 5-seed ensemble oracle from %s", cfg["oracle_ckpt_dir"])
    oracle = EnsembleForwardOracle(cfg["oracle_ckpt_dir"])

    # Proposer
    if args.no_llm:
        seed = args.seed if args.seed is not None else cfg.get("seed", 0)
        proposer = _RandomOnlyProposer(seed=seed)
        logger.info("Using random-only proposer (--no-llm)")
    else:
        proposer_cfg = cfg.get("proposer", {})
        model_name = (proposer_cfg.get("fallback_model") if args.use_fallback_llm
                      else proposer_cfg.get("model_name", "Qwen/Qwen2.5-VL-3B-Instruct"))
        logger.info("Loading Qwen proposer: %s", model_name)
        proposer = QwenProposer(
            model_name=model_name,
            max_new_tokens=proposer_cfg.get("max_new_tokens", 512),
            temperature=proposer_cfg.get("temperature", 0.7),
            top_p=proposer_cfg.get("top_p", 0.9),
            seed=args.seed if args.seed is not None else cfg.get("seed", 0),
        )

    # Run all targets
    summary_rows = []
    for tbg in targets:
        records = run_one_target(tbg, cfg, args, df, oracle, proposer,
                                 a_vocab, b_vocab, out_root)
        if records:
            best = min(records, key=lambda r: r.error_to_target)
            summary_rows.append({
                "target_bg": tbg,
                "best_pred_bg_mean": best.pred_bg_mean,
                "best_pred_bg_std": best.pred_bg_std,
                "best_error": best.error_to_target,
                "best_A": best.A, "best_B": best.B, "best_base": best.base_mid,
                "best_charge_warning": best.charge_warning,
                "best_low_confidence": best.low_confidence,
            })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_root / "summary.csv", index=False)
        logger.info("Summary written:\n%s", summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
