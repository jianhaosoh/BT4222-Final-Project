"""Compare per-request shared evaluation outputs on identical requests."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import read_table


def user_cluster_ci(frame: pd.DataFrame, metric: str, *, reps: int = 2000, seed: int = 42) -> dict:
    valid = frame.loc[frame[metric].notna(), ["user_id", metric]]
    values = pd.to_numeric(valid[metric], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{metric} contains infinite values")
    grouped = valid.groupby("user_id", sort=False)[metric].agg(["sum", "count"])
    if len(grouped) < 2:
        return {"estimate": float(valid[metric].mean()) if len(valid) else None,
                "ci95": None, "users": int(len(grouped)), "requests": int(len(valid)),
                "missing_requests": int(len(frame) - len(valid))}
    sums, counts = grouped["sum"].to_numpy(float), grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), size=(reps, len(sums)))
    sample = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"estimate": float(sums.sum() / counts.sum()),
            "ci95": [float(np.quantile(sample, .025)), float(np.quantile(sample, .975))],
            "users": int(len(grouped)), "requests": int(len(valid)),
            "missing_requests": int(len(frame) - len(valid))}


def paired_ci(a: pd.DataFrame, b: pd.DataFrame, metric: str, *, reps: int = 2000, seed: int = 42) -> dict:
    paired = a[["behaviour_id", "user_id", metric]].merge(
        b[["behaviour_id", metric]], on="behaviour_id", suffixes=("_a", "_b"),
        validate="one_to_one",
    ).dropna(subset=[f"{metric}_a", f"{metric}_b"])
    paired["delta"] = paired[f"{metric}_b"] - paired[f"{metric}_a"]
    grouped = paired.groupby("user_id", sort=False).delta.agg(["sum", "count"])
    if len(grouped) < 2:
        return {"delta_b_minus_a": float(paired.delta.mean()) if len(paired) else None,
                "ci95": None, "users": int(len(grouped)), "paired_requests": int(len(paired)),
                "availability_mismatch_requests": int(a[metric].notna().ne(b[metric].notna()).sum())}
    sums, counts = grouped["sum"].to_numpy(float), grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), size=(reps, len(sums)))
    sample = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"delta_b_minus_a": float(sums.sum() / counts.sum()),
            "ci95": [float(np.quantile(sample, .025)), float(np.quantile(sample, .975))],
            "users": int(len(grouped)), "paired_requests": int(len(paired)),
            "availability_mismatch_requests": int(a[metric].notna().ne(b[metric].notna()).sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PER_REQUEST_FILE",
                        help="Repeat for each model's per-request metrics output")
    parser.add_argument("--reference", required=True, help="Model name used as paired comparison reference")
    parser.add_argument("--metrics", nargs="+", default=["known_click_recall@100",
                        "observed_click_ndcg@100", "known_click_recall@200", "observed_click_ndcg@200"])
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    if not args.metrics:
        parser.error("--metrics must contain at least one metric")
    if len(args.metrics) != len(set(args.metrics)):
        parser.error("--metrics must not contain duplicates")
    models = {}
    shared_diversity_requested = any(
        metric.startswith("shared_intra_list_diversity@") for metric in args.metrics
    )
    for spec in args.model:
        if "=" not in spec:
            parser.error("Each --model must be NAME=PATH")
        name, filename = spec.split("=", 1)
        if not name.strip() or not filename.strip():
            parser.error("Each --model must have a non-empty NAME and PATH")
        name = name.strip()
        if name in models:
            parser.error(f"Duplicate model name: {name}")
        frame = read_table(Path(filename))
        required = {"behaviour_id", "user_id", "split", "eligibility_policy",
                    "metric_definition_version", "naive_timezone_policy", "topic_fallback_policy",
                    "export_k", "metric_cutoffs_json",
                    "catalogue_id_fingerprint_sha256", "catalogue_content_fingerprint_sha256",
                    "request_id_fingerprint_sha256", *args.metrics}
        if shared_diversity_requested:
            required.add("similarity_constants_sha256")
        if required - set(frame):
            raise ValueError(f"{name} per-request file missing {sorted(required - set(frame))}")
        export_values = pd.to_numeric(frame.export_k, errors="raise").dropna().unique()
        cutoff_values = frame.metric_cutoffs_json.dropna().astype(str).unique()
        if len(export_values) != 1 or len(cutoff_values) != 1:
            raise ValueError(f"{name} must have one non-null export_k and metric_cutoffs_json")
        cutoffs = set(json.loads(cutoff_values[0]))
        for metric in args.metrics:
            match = re.search(r"@(\d+)$", metric)
            if match and int(match.group(1)) > int(export_values[0]):
                raise ValueError(f"{name} export_k is too small to support {metric}")
            if match and int(match.group(1)) not in cutoffs:
                raise ValueError(f"{name} per-request file was not evaluated at cutoff {match.group(1)}")
        for metric in args.metrics:
            frame[metric] = pd.to_numeric(frame[metric], errors="raise")
            values = frame[metric].dropna().to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{name}.{metric} contains infinite values")
            bounded_metric_prefixes = (
                "known_click_recall@", "eligible_only_recall@", "observed_click_ndcg@",
                "topic_article_coverage@", "unknown_age_share@", "age_le_24h_share@",
                "age_24_72h_share@", "age_gt_72h_share@", "intra_list_diversity@",
                "shared_intra_list_diversity@",
                "has_text_embedding_share@", "has_image_embedding_share@",
                "has_text_embedding_missing_share@", "has_image_embedding_missing_share@",
                "has_text_embedding_unknown_share@", "has_image_embedding_unknown_share@",
                "observed_continuation_recall@", "continuation_observation_coverage@",
                "continuation_rate_observed_topk@", "session_continuity_proxy@",
                "continuity_proxy_coverage@",
            )
            if metric.startswith(bounded_metric_prefixes):
                if ((values < 0) | (values > 1)).any():
                    raise ValueError(f"{name}.{metric} must be within [0,1]")
        frame = frame.sort_values("behaviour_id").reset_index(drop=True)
        if frame.behaviour_id.duplicated().any():
            raise ValueError(f"{name} has duplicate behaviour_id rows")
        models[name] = frame
    if args.reference not in models:
        parser.error(f"Reference model {args.reference!r} was not supplied")
    reference = models[args.reference]
    ref_keys = reference.behaviour_id.tolist()
    for name, frame in models.items():
        if frame.behaviour_id.tolist() != ref_keys:
            raise ValueError(f"{name} and {args.reference} do not cover identical validation requests")
        if not frame.user_id.equals(reference.user_id):
            raise ValueError(f"{name} has inconsistent user_id mapping for shared requests")
        for column in ("split", "eligibility_policy", "metric_definition_version",
                       "naive_timezone_policy", "topic_fallback_policy", "catalogue_id_fingerprint_sha256",
                       "catalogue_content_fingerprint_sha256",
                       "request_id_fingerprint_sha256"):
            if (column in reference) != (column in frame):
                raise ValueError(f"{name} and reference do not both record {column}")
            if column in frame:
                values = frame[column].dropna().astype(str).unique()
                reference_values = reference[column].dropna().astype(str).unique()
                if len(values) != 1 or len(reference_values) != 1:
                    raise ValueError(f"Each model must have exactly one non-null {column}")
                if values[0] != reference_values[0]:
                    raise ValueError(f"{name} and reference differ in {column}")
        if shared_diversity_requested:
            column = "similarity_constants_sha256"
            values = frame[column].dropna().astype(str).unique()
            reference_values = reference[column].dropna().astype(str).unique()
            if len(values) != 1 or len(reference_values) != 1 or values[0] != reference_values[0]:
                raise ValueError(f"{name} and reference differ in {column}")

    metadata_fields = ("split", "eligibility_policy", "metric_definition_version", "naive_timezone_policy",
                       "topic_fallback_policy", "export_k", "metric_cutoffs_json",
                       "catalogue_id_fingerprint_sha256", "catalogue_content_fingerprint_sha256",
                       "request_id_fingerprint_sha256", "similarity_constants_sha256", "run_id")
    metadata = {}
    for model, frame in models.items():
        metadata[model] = {}
        for col in metadata_fields:
            value = frame[col].dropna().iloc[0] if col in frame and frame[col].notna().any() else None
            metadata[model][col] = value.item() if hasattr(value, "item") else value
    report = {"request_count": len(ref_keys), "user_count": int(reference.user_id.nunique()),
              "point_estimate_weighting": "request-weighted; users are resampled as clusters",
              "reference_model": args.reference, "model_metadata": metadata, "metrics": {}}
    for metric in args.metrics:
        report["metrics"][metric] = {
            "models": {name: user_cluster_ci(frame, metric, reps=args.replicates, seed=args.seed)
                       for name, frame in models.items()},
            "paired_vs_reference": {
                name: paired_ci(reference, frame, metric, reps=args.replicates, seed=args.seed)
                for name, frame in models.items() if name != args.reference
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
