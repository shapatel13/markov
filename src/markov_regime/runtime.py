from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RunProfile = Literal["core_signal", "full_research"]


@dataclass(frozen=True)
class AnalysisPlan:
    profile: RunProfile
    run_model_comparison: bool
    run_robustness: bool
    run_timeframe_comparison: bool
    run_feature_pack_comparison: bool
    run_consensus_diagnostics: bool
    run_candidate_search: bool
    state_values: tuple[int, ...]
    note: str


def resolve_analysis_plan(
    *,
    profile: RunProfile,
    selected_states: int,
    run_model_comparison: bool,
    run_robustness: bool,
    run_timeframe_comparison: bool,
    run_feature_pack_comparison: bool,
    run_consensus_diagnostics: bool,
    run_candidate_search: bool,
    require_consensus_confirmation: bool,
    engine_mode: str,
) -> AnalysisPlan:
    consensus_required = bool(require_consensus_confirmation or engine_mode == "hmm_ensemble")

    if profile == "core_signal":
        return AnalysisPlan(
            profile="core_signal",
            run_model_comparison=False,
            run_robustness=False,
            run_timeframe_comparison=False,
            run_feature_pack_comparison=False,
            run_consensus_diagnostics=consensus_required,
            run_candidate_search=False,
            state_values=(int(selected_states),),
            note=(
                "Core signal mode evaluates only the selected HMM state count and skips optional research "
                "diagnostics for speed. Consensus still runs automatically if the chosen engine or guardrails require it."
            ),
        )

    effective_model_comparison = bool(run_model_comparison)
    state_values = tuple(range(5, 10)) if effective_model_comparison else (int(selected_states),)
    return AnalysisPlan(
        profile="full_research",
        run_model_comparison=effective_model_comparison,
        run_robustness=bool(run_robustness),
        run_timeframe_comparison=bool(run_timeframe_comparison),
        run_feature_pack_comparison=bool(run_feature_pack_comparison),
        run_consensus_diagnostics=bool(run_consensus_diagnostics or consensus_required),
        run_candidate_search=bool(run_candidate_search),
        state_values=state_values,
        note=(
            "Full research mode honors the slower diagnostics you selected, including state-count comparison, "
            "cross-asset robustness, feature ablations, and candidate search."
        ),
    )
