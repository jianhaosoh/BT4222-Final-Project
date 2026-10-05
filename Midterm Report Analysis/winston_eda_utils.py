from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    PowerTransformer,
    SplineTransformer,
    StandardScaler,
)


COLD_START_FEATURE_TYPES = {
    "article_age_hours": "continuous",
    "position": "ordinal",
    "slate_size": "count",
    "title_length": "count",
    "body_length": "count",
    "entity_count": "count",
    "image_count": "count",
    "topic_count": "count",
    "premium": "binary",
    "category": "categorical",
    "clicked": "binary",
}

COLD_START_DISPLAY_TRANSFORMS = {
    "article_age_hours": "log1p",
    "position": "log1p",
    "slate_size": "log1p",
    "body_length": "log1p",
    "entity_count": "log1p",
    "image_count": "log1p",
}


def distribution_frame_from_maps(feature_maps: Mapping[str, Mapping]) -> pd.DataFrame:
    """Convert DuckDB histogram maps into compact weighted plotting data."""
    rows = [
        {"feature": feature, "value": value, "count": count, "plot_label": feature}
        for feature, counts in feature_maps.items()
        for value, count in counts.items()
    ]
    return pd.DataFrame(rows)


def transform_distribution_frame(
    distribution: pd.DataFrame,
    transforms: Mapping[str, str],
) -> pd.DataFrame:
    """Apply one-dimensional display transforms while preserving frequencies."""
    result = distribution.copy()
    for feature, method in transforms.items():
        mask = result["feature"].eq(feature)
        values = pd.to_numeric(result.loc[mask, "value"])
        if method == "log1p":
            if (values < 0).any():
                raise ValueError(f"log1p requires non-negative values: {feature}")
            result.loc[mask, "value"] = np.log1p(values)
        elif method == "sqrt":
            if (values < 0).any():
                raise ValueError(f"sqrt requires non-negative values: {feature}")
            result.loc[mask, "value"] = np.sqrt(values)
        elif method != "raw":
            raise ValueError(f"Unknown display transformation: {method}")
        result.loc[mask, "plot_label"] = f"{method}({feature})"
    return result


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(sorted_values[np.searchsorted(cumulative, q * cumulative[-1])])


def plot_distribution_matrix(
    distribution: pd.DataFrame,
    feature_types: Mapping[str, str],
    *,
    title: str,
):
    """Plot weighted feature distributions in a five-column matrix."""
    import matplotlib.pyplot as plt

    n_columns = 5
    n_rows = int(np.ceil(len(feature_types) / n_columns))
    figure, axes = plt.subplots(
        n_rows, n_columns, figsize=(18, 3.2 * n_rows), squeeze=False
    )
    for axis, (feature, kind) in zip(axes.flat, feature_types.items()):
        data = distribution.loc[distribution["feature"].eq(feature)].copy()
        label = data["plot_label"].iloc[0]
        if kind in {"categorical", "binary"}:
            counts = data.groupby("value", dropna=False)["count"].sum().nlargest(12)
            shares = 100 * counts / data["count"].sum()
            axis.barh(shares.index.astype(str), shares, color="#4C78A8")
            axis.set_xlabel("Candidates (%)")
        else:
            values = pd.to_numeric(data["value"]).to_numpy(float)
            weights = data["count"].to_numpy(float)
            upper = _weighted_quantile(values, weights, 0.995)
            shown = values <= upper
            axis.hist(values[shown], bins=30, weights=100 * weights[shown] / weights.sum(), color="#4C78A8")
            axis.set_ylabel("Candidates (%)")
            axis.text(0.98, 0.92, "central 99.5%", ha="right", transform=axis.transAxes, fontsize=8)
        axis.set_title(f"{label}\n{kind}", fontsize=10)
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(feature_types):]:
        axis.set_visible(False)
    figure.suptitle(title, fontsize=15, y=1.01)
    figure.tight_layout()
    return figure, axes


def ranking_metrics(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    groups: pd.Series | np.ndarray,
    *,
    k: int = 10,
) -> dict[str, float]:
    """Calculate binary and impression-grouped ranking metrics."""
    labels = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(y_score, dtype=float)
    group_values = np.asarray(groups)

    if not (len(labels) == len(scores) == len(group_values)):
        raise ValueError("Labels, scores, and groups must have equal lengths.")
    if len(labels) == 0:
        raise ValueError("Metrics require at least one observation.")

    clipped = np.clip(scores, 1e-12, 1 - 1e-12)
    result = {
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        "pr_auc": float(average_precision_score(labels, scores))
        if labels.sum() > 0
        else np.nan,
        "roc_auc": float(roc_auc_score(labels, scores))
        if np.unique(labels).size == 2
        else np.nan,
    }

    ranked = pd.DataFrame(
        {"group": group_values, "clicked": labels, "score": scores}
    ).sort_values(["group", "score"], ascending=[True, False], kind="mergesort")
    ranked["rank"] = ranked.groupby("group", sort=False).cumcount() + 1

    discounts = 1 / np.log2(np.arange(2, k + 2))
    ranked["discounted_gain"] = np.where(
        ranked["rank"] <= k,
        ranked["clicked"] / np.log2(ranked["rank"] + 1),
        0.0,
    )
    dcg = ranked.groupby("group", sort=False)["discounted_gain"].sum()
    clicks = ranked.groupby("group", sort=False)["clicked"].sum().astype(int)
    ideal_cumulative = np.cumsum(discounts)
    idcg = clicks.clip(upper=k).map(
        lambda count: ideal_cumulative[count - 1] if count > 0 else np.nan
    )
    result[f"ndcg_at_{k}"] = float((dcg / idcg).dropna().mean())

    first_click = ranked.loc[ranked["clicked"].eq(1)].groupby("group")["rank"].min()
    result["mrr"] = float((1 / first_click).mean()) if len(first_click) else np.nan
    return result


def calibration_table(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    """Summarize observed and predicted click rates by score quantile."""
    frame = pd.DataFrame(
        {"clicked": np.asarray(y_true), "score": np.asarray(y_score, dtype=float)}
    )
    if len(frame) == 0:
        raise ValueError("Calibration requires at least one observation.")

    frame["score_bin"] = pd.qcut(
        frame["score"], q=min(bins, len(frame)), labels=False, duplicates="drop"
    )
    summary = (
        frame.groupby("score_bin", observed=True)
        .agg(
            predicted_click_rate=("score", "mean"),
            observed_click_rate=("clicked", "mean"),
            observations=("clicked", "size"),
        )
        .reset_index()
    )
    summary["score_bin"] = summary["score_bin"] + 1
    return summary


def numeric_vif(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate variance inflation factors for a numeric feature frame."""
    if frame.shape[1] == 0:
        raise ValueError("VIF requires at least one feature.")

    values = frame.astype(float).replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median())
    if values.shape[1] == 1:
        return pd.DataFrame({"feature": values.columns, "vif": [1.0]})

    rows: list[dict[str, float | str]] = []
    for feature in values.columns:
        target = values[feature]
        predictors = values.drop(columns=feature)
        r_squared = LinearRegression().fit(predictors, target).score(predictors, target)
        vif = np.inf if r_squared >= 1 - 1e-12 else 1 / (1 - r_squared)
        rows.append({"feature": feature, "vif": float(vif)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False, ignore_index=True)


def fit_logistic_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    numeric_transforms: Mapping[str, str],
    categorical: Sequence[str] = (),
    target: str = "clicked",
    group: str = "impression_id",
    penalty: str = "l2",
    c: float = 1.0,
    l1_ratio: float | None = None,
    class_weight: str | None = None,
    max_iter: int = 400,
    tol: float = 1e-4,
) -> tuple[Pipeline, np.ndarray, dict[str, float]]:
    """Fit a leakage-safe preprocessing and logistic-regression pipeline."""
    transformers: list[tuple[str, Any, list[str]]] = []
    for feature, method in numeric_transforms.items():
        transformers.append((feature, _numeric_pipeline(method), [feature]))

    if categorical:
        category_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    OneHotEncoder(handle_unknown="ignore", min_frequency=50),
                ),
            ]
        )
        transformers.append(("categorical", category_pipeline, list(categorical)))

    preprocess = ColumnTransformer(transformers, remainder="drop")
    if penalty == "l2":
        solver = "lbfgs"
    elif penalty == "l1":
        solver = "liblinear"
    else:
        solver = "saga"
    model = LogisticRegression(
        C=c,
        penalty=penalty,
        solver=solver,
        l1_ratio=l1_ratio,
        class_weight=class_weight,
        max_iter=max_iter,
        tol=tol,
        random_state=42,
    )
    pipeline = Pipeline([("preprocess", preprocess), ("model", model)])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Bins whose width are too small.*",
            category=UserWarning,
        )
        pipeline.fit(train, train[target].astype(int))
    scores = pipeline.predict_proba(valid)[:, 1]
    metrics = ranking_metrics(valid[target], scores, valid[group], k=10)
    return pipeline, scores, metrics


def screen_numeric_transforms(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    feature: str,
    controls: Sequence[str],
    methods: Sequence[str] = ("raw", "log1p", "sqrt", "yeo-johnson", "binned", "spline"),
    target: str = "clicked",
    group: str = "impression_id",
) -> pd.DataFrame:
    """Compare alternative representations of one numeric feature."""
    rows: list[dict[str, float | int | str]] = []
    control_features = [name for name in controls if name != feature]

    for method in methods:
        transforms = {name: "raw" for name in control_features}
        transforms[feature] = method
        pipeline, _, metrics = fit_logistic_model(
            train,
            valid,
            numeric_transforms=transforms,
            target=target,
            group=group,
        )
        width = pipeline.named_steps["preprocess"].transform(valid.iloc[:1]).shape[1]
        rows.append(
            {
                "feature": feature,
                "transform": method,
                "model_columns": width,
                **metrics,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["log_loss", "ndcg_at_10"], ascending=[True, False], ignore_index=True
    )


def _numeric_pipeline(method: str) -> Pipeline:
    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]

    if method == "raw":
        steps.append(("scale", StandardScaler()))
    elif method == "log1p":
        steps.extend(
            [
                ("transform", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]
        )
    elif method == "sqrt":
        steps.extend(
            [
                ("transform", FunctionTransformer(np.sqrt, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]
        )
    elif method == "yeo-johnson":
        steps.append(("transform", PowerTransformer(method="yeo-johnson")))
    elif method == "binned":
        steps.append(
            (
                "transform",
                KBinsDiscretizer(
                    n_bins=8,
                    encode="onehot",
                    strategy="quantile",
                    quantile_method="averaged_inverted_cdf",
                ),
            )
        )
    elif method == "spline":
        steps.extend(
            [
                ("transform", SplineTransformer(n_knots=5, degree=3, include_bias=False)),
                ("scale", StandardScaler(with_mean=False)),
            ]
        )
    else:
        raise ValueError(f"Unknown numeric transformation: {method}")
    return Pipeline(steps)
