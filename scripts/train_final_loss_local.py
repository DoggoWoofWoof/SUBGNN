"""Run final-loss Jigsaw training without launching Modal.

This is for Lightning/remote Linux shells where we want the same training code
but executed on the current GPU. The heavy training function still lives in
`scripts.modal_train_graphsage`; this wrapper calls its Modal `.local(...)`
method so checkpoint/model paths remain compatible with the existing pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.modal_train_graphsage import train


def _upload_lightning_results(cache_root: str, reason: str) -> None:
    model_name = os.environ.get("LIGHTNING_RESULTS_MODEL", "").strip()
    if not model_name:
        return
    try:
        from lightning_sdk.models import upload_model

        cache_path = Path(cache_root).resolve()
        if not cache_path.exists():
            return
        print(f"[LIGHTNING] Uploading {reason} artifacts from {cache_path} to {model_name}", flush=True)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
            upload_model(model_name, path=cache_path, progress_bar=False)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
        print(f"[LIGHTNING] Upload complete ({reason}).", flush=True)
    except Exception as exc:
        print(f"[LIGHTNING] Upload failed ({reason}, non-fatal): {exc}", flush=True)


def _ensure_cache(cache_root: str) -> None:
    cache_path = Path(cache_root)
    cache_path.mkdir(parents=True, exist_ok=True)
    (cache_path / "logs").mkdir(parents=True, exist_ok=True)
    (cache_path / "models").mkdir(parents=True, exist_ok=True)

    # The shared training code is currently hard-wired to /cache. On Linux,
    # allow users to keep persistent data elsewhere while preserving paths.
    if os.name != "nt" and cache_root != "/cache" and not Path("/cache").exists():
        os.symlink(cache_path, "/cache", target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local/Lightning Jigsaw final-loss trainer")
    parser.add_argument("--dataset", choices=["cora", "arxiv", "mag"], default="mag")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--training-seed", type=int, default=7202)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--cache-root", default="/cache")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--resume-model-only", action="store_true",
                        help="Load only encoder weights from the checkpoint (fresh optimizer/schedule/epoch) "
                             "-- use when fine-tuning from a prior model on a new query mix.")
    parser.add_argument("--encoder-kind", choices=["graphsage", "rgcn"], default="rgcn")
    parser.add_argument("--momentum-cache-decay", type=float, default=0.99)
    parser.add_argument("--coverage-target-mode", choices=["hard", "overlap", "overlap_union"], default="overlap")
    parser.add_argument("--coverage-topk", type=int, default=50)
    parser.add_argument("--coverage-topk-bucket-size", type=int, default=10)
    parser.add_argument("--coverage-topk-weight", type=float, default=0.35)
    parser.add_argument("--coverage-cvar-fraction", type=float, default=0.5)
    parser.add_argument("--max-live-positive-parts", type=int, default=64)
    parser.add_argument("--max-train-coarse-parts", type=int, default=80)
    parser.add_argument("--cache-refresh-steps", type=int, default=10)
    parser.add_argument("--cache-encode-batch-size", type=int, default=32)
    parser.add_argument("--cache-partition-graphs", type=int, default=1)
    parser.add_argument("--query-target-sizes", default="20,50,50,100,100")
    parser.add_argument("--query-size-jitter", type=int, default=5)
    parser.add_argument("--prob-k-hop", type=float, default=0.35)
    parser.add_argument("--prob-single-part", type=float, default=0.10)
    parser.add_argument("--prob-multi-coarse", type=float, default=0.25)
    parser.add_argument("--prob-random-walk", type=float, default=0.15)
    parser.add_argument("--prob-degree-k-hop", type=float, default=0.10)
    parser.add_argument("--validation-queries", type=int, default=50)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--validation-seeds", default="31415,27182")
    parser.add_argument("--validation-topks", default="20,50,100,200,500,1000")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--min-learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--checkpoint-interval-epochs", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop if validation FullCov does not improve for this many checks (0=off). "
                             "The validation-best checkpoint is always kept regardless.")
    args = parser.parse_args()

    _ensure_cache(args.cache_root)

    run_name = args.run_name
    if not run_name:
        run_name = (
            f"{args.encoder_kind}_final_loss_{args.dataset}_seed{args.training_seed}_"
            f"{args.coverage_target_mode}_topk{args.coverage_topk}_live{args.max_live_positive_parts}"
        )
    checkpoint_path = f"/cache/{args.dataset}_{run_name}_checkpoint.pth"

    result = train.local(
        dataset_name=args.dataset,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        num_hierarchies=1,
        checkpoint_path=checkpoint_path,
        fresh_start=args.fresh,
        run_name=run_name,
        gamma_partition=1.0,
        coverage_temperature=0.05,
        coverage_topk=args.coverage_topk,
        coverage_topk_bucket_size=args.coverage_topk_bucket_size,
        coverage_topk_weight=args.coverage_topk_weight,
        coverage_topk_margin=0.0,
        coverage_positive_aggregation="cvar",
        coverage_cvar_fraction=args.coverage_cvar_fraction,
        coverage_smoothmax_temperature=0.1,
        max_live_positive_parts=args.max_live_positive_parts,
        gamma_fine_partition=0.0,
        fine_cache_refresh_steps=250,
        alpha=0.2,
        beta=0.0,
        prob_k_hop=args.prob_k_hop,
        prob_single_part=args.prob_single_part,
        prob_multi_coarse=args.prob_multi_coarse,
        prob_random_walk=args.prob_random_walk,
        prob_degree_k_hop=args.prob_degree_k_hop,
        hard_negative_source="cache" if args.dataset == "mag" else "graphs",
        max_gpos_nodes=2500,
        max_train_coarse_parts=args.max_train_coarse_parts,
        query_target_sizes=args.query_target_sizes,
        query_size_jitter=args.query_size_jitter,
        cache_refresh_steps=args.cache_refresh_steps,
        cache_encode_batch_size=args.cache_encode_batch_size,
        cache_partition_graphs=args.cache_partition_graphs,
        checkpoint_interval_epochs=args.checkpoint_interval_epochs,
        resume_from_checkpoint=args.resume_from_checkpoint,
        learning_rate=args.learning_rate,
        scheduler_type="cosine",
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        plateau_patience=10,
        plateau_factor=0.5,
        cosine_t_max=0,
        resume_model_only=args.resume_model_only,
        validation_queries=args.validation_queries,
        validation_interval=args.validation_interval,
        validation_seed=31415,
        validation_seeds=args.validation_seeds,
        validation_topks=args.validation_topks,
        early_stopping_patience=args.early_stopping_patience,
        training_seed=args.training_seed,
        disable_residual=False,
        encoder_kind=args.encoder_kind,
        momentum_cache_decay=args.momentum_cache_decay,
        coverage_target_mode=args.coverage_target_mode,
    )
    _upload_lightning_results(args.cache_root, "train-end")
    print(result, flush=True)


if __name__ == "__main__":
    main()
