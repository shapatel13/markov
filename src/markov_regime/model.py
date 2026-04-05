from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

from markov_regime.config import ModelConfig

SIGNATURE_COLUMNS: tuple[str, ...] = (
    "mean_return",
    "volatility",
    "persistence",
    "frequency",
)


@dataclass
class FittedHMM:
    model: GaussianHMM
    scaler: StandardScaler
    converged: bool
    fit_messages: tuple[str, ...] = ()


def fit_hmm(
    train_frame: pd.DataFrame,
    feature_columns: Iterable[str],
    config: ModelConfig,
) -> FittedHMM:
    x_train = train_frame.loc[:, list(feature_columns)].to_numpy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    model = GaussianHMM(
        n_components=config.n_states,
        covariance_type=config.covariance_type,
        n_iter=config.n_iter,
        random_state=config.random_state,
        min_covar=config.min_covar,
    )
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stderr(stderr_buffer):
        model.fit(x_scaled)
    fit_messages = tuple(line.strip() for line in stderr_buffer.getvalue().splitlines() if line.strip())
    converged = bool(getattr(model.monitor_, "converged", True))
    return FittedHMM(model=model, scaler=scaler, converged=converged, fit_messages=fit_messages)


def annotate_posteriors(
    frame: pd.DataFrame,
    fitted: FittedHMM,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    x_values = frame.loc[:, list(feature_columns)].to_numpy()
    x_scaled = fitted.scaler.transform(x_values)
    probabilities = fitted.model.predict_proba(x_scaled)
    raw_states = probabilities.argmax(axis=1)
    ordered_probabilities = np.sort(probabilities, axis=1)
    annotated = frame.copy()
    annotated["raw_state"] = raw_states
    annotated["max_posterior"] = ordered_probabilities[:, -1]
    annotated["second_posterior"] = ordered_probabilities[:, -2] if probabilities.shape[1] > 1 else 0.0
    annotated["confidence_gap"] = annotated["max_posterior"] - annotated["second_posterior"]
    annotated["posterior_entropy"] = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1)
    for state in range(probabilities.shape[1]):
        annotated[f"posterior_{state}"] = probabilities[:, state]
    return annotated


def summarize_states(
    frame: pd.DataFrame,
    state_column: str = "raw_state",
    edge_horizon: int = 6,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    edge_column = f"forward_return_{edge_horizon}"
    all_states = sorted(int(state) for state in frame[state_column].dropna().unique())
    for state in all_states:
        mask = frame[state_column] == state
        subset = frame.loc[mask]
        mask_values = mask.to_numpy(dtype=bool)
        persistence_denominator = int(mask_values[:-1].sum()) if len(mask_values) > 1 else 0
        persistence_numerator = int(np.logical_and(mask_values[:-1], mask_values[1:]).sum()) if len(mask_values) > 1 else 0
        persistence = float(persistence_numerator / persistence_denominator) if persistence_denominator else 0.0
        rows.append(
            {
                "state": state,
                "samples": int(mask.sum()),
                "mean_return": float(subset["bar_return"].mean()),
                "volatility": float(subset["bar_return"].std(ddof=0) if len(subset) > 1 else 0.0),
                "persistence": persistence,
                "frequency": float(mask.mean()),
                "avg_confidence": float(subset["max_posterior"].mean() if "max_posterior" in subset else 0.0),
                "validation_edge": float(subset[edge_column].mean() if edge_column in subset else 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("state").reset_index(drop=True)


def initial_state_mapping(summary: pd.DataFrame) -> dict[int, int]:
    ordering = summary.sort_values(["validation_edge", "mean_return", "state"]).reset_index(drop=True)
    return {int(row.state): int(index) for index, row in ordering.iterrows()}


def align_state_mapping(
    reference_summary: pd.DataFrame,
    current_summary: pd.DataFrame,
) -> tuple[dict[int, int], pd.DataFrame]:
    ref = reference_summary.sort_values("canonical_state").reset_index(drop=True)
    cur = current_summary.sort_values("state").reset_index(drop=True)

    ref_values = ref.loc[:, list(SIGNATURE_COLUMNS)].fillna(0.0).to_numpy()
    cur_values = cur.loc[:, list(SIGNATURE_COLUMNS)].fillna(0.0).to_numpy()
    scale = np.nanstd(np.vstack([ref_values, cur_values]), axis=0)
    scale[scale == 0.0] = 1.0
    normalized_ref = ref_values / scale
    normalized_cur = cur_values / scale

    cost_matrix = cdist(normalized_ref, normalized_cur, metric="euclidean")
    row_index, column_index = linear_sum_assignment(cost_matrix)

    mapping = {
        int(cur.iloc[current_row]["state"]): int(ref.iloc[reference_row]["canonical_state"])
        for reference_row, current_row in zip(row_index, column_index, strict=True)
    }
    alignment = pd.DataFrame(
        {
            "canonical_state": [int(ref.iloc[row]["canonical_state"]) for row in row_index],
            "raw_state": [int(cur.iloc[column]["state"]) for column in column_index],
            "alignment_distance": [float(cost_matrix[row, column]) for row, column in zip(row_index, column_index, strict=True)],
        }
    )
    return mapping, alignment


def apply_state_mapping(frame: pd.DataFrame, mapping: dict[int, int]) -> pd.DataFrame:
    aligned = frame.copy()
    aligned["canonical_state"] = aligned["raw_state"].map(mapping).astype(int)
    inverse_mapping = {canonical_state: raw_state for raw_state, canonical_state in mapping.items()}
    for canonical_state, raw_state in inverse_mapping.items():
        aligned[f"canonical_posterior_{canonical_state}"] = aligned[f"posterior_{raw_state}"]
    return aligned


def map_summary(summary: pd.DataFrame, mapping: dict[int, int]) -> pd.DataFrame:
    mapped = summary.copy()
    mapped["canonical_state"] = mapped["state"].map(mapping).astype(int)
    return mapped.sort_values("canonical_state").reset_index(drop=True)


def blend_reference_summary(
    previous_reference: pd.DataFrame | None,
    current_summary: pd.DataFrame,
    weight: float = 0.5,
) -> pd.DataFrame:
    if previous_reference is None:
        return current_summary.copy()

    previous = previous_reference.set_index("canonical_state")
    current = current_summary.set_index("canonical_state")
    blended = current.copy()
    for column in SIGNATURE_COLUMNS:
        blended[column] = (1.0 - weight) * previous[column] + weight * current[column]
    return blended.reset_index()


def information_criteria(
    fitted: FittedHMM,
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
) -> tuple[float, float, float]:
    x_values = frame.loc[:, list(feature_columns)].to_numpy()
    x_scaled = fitted.scaler.transform(x_values)
    log_likelihood = float(fitted.model.score(x_scaled))
    n_states = fitted.model.n_components
    n_features = x_scaled.shape[1]

    transition_params = n_states * (n_states - 1)
    start_params = n_states - 1
    mean_params = n_states * n_features
    covariance_params = n_states * (n_features * (n_features + 1) / 2)
    parameter_count = transition_params + start_params + mean_params + covariance_params

    aic = float(2.0 * parameter_count - 2.0 * log_likelihood)
    bic = float(np.log(len(x_scaled)) * parameter_count - 2.0 * log_likelihood)
    return log_likelihood, aic, bic
