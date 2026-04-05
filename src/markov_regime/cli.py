from __future__ import annotations

import argparse
from dataclasses import replace

from markov_regime.confirmation import apply_higher_timeframe_confirmation
from markov_regime.consensus import apply_consensus_confirmation, run_consensus_diagnostics
from markov_regime.config import (
    DataConfig,
    ModelConfig,
    StrategyConfig,
    SweepConfig,
    WalkForwardConfig,
    default_walk_forward_config,
)
from markov_regime.data import fetch_price_data
from markov_regime.features import build_feature_frame, get_feature_columns, list_feature_packs
from markov_regime.readiness import DEFAULT_AUDIT_STRATEGY, run_primetime_audit, write_primetime_audit_report
from markov_regime.research import (
    ensure_results_tsv,
    load_research_program,
    run_candidate_search,
    run_autoresearch,
    run_feature_pack_comparison,
    run_timeframe_comparison,
    summarize_candidate_search,
    write_research_program,
)
from markov_regime.reporting import export_signal_report
from markov_regime.robustness import parse_symbol_list
from markov_regime.strategy import parameter_sweep
from markov_regime.walkforward import run_walk_forward, suggest_walk_forward_config

DEFAULT_CLI_INTERVAL = "4hour"


def _parse_csv_strings(values: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in values.split(",") if item.strip())


def _parse_csv_ints(values: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in values.split(",") if item.strip())


def _parse_short_modes(values: str) -> tuple[bool, ...]:
    parsed: list[bool] = []
    mapping = {"on": True, "off": False, "true": True, "false": False, "long_only": False, "long_short": True}
    for item in values.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"Unsupported short mode: {item}")
        parsed.append(mapping[key])
    return tuple(parsed)


def _resolve_cli_walk_config(args: argparse.Namespace) -> WalkForwardConfig:
    interval_defaults = default_walk_forward_config(args.interval)
    return WalkForwardConfig(
        train_bars=args.train_bars if args.train_bars is not None else interval_defaults.train_bars,
        purge_bars=args.purge_bars if args.purge_bars is not None else interval_defaults.purge_bars,
        validate_bars=args.validate_bars if args.validate_bars is not None else interval_defaults.validate_bars,
        embargo_bars=args.embargo_bars if args.embargo_bars is not None else interval_defaults.embargo_bars,
        test_bars=args.test_bars if args.test_bars is not None else interval_defaults.test_bars,
        refit_stride_bars=args.refit_stride_bars if args.refit_stride_bars is not None else interval_defaults.refit_stride_bars,
    )


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--interval", choices=["4hour", "1day", "1hour"], default=DEFAULT_CLI_INTERVAL)
    parser.add_argument("--provider", choices=["auto", "fmp", "coinbase", "yahoo"], default="auto")
    parser.add_argument("--feature-pack", choices=list(list_feature_packs()), default="mean_reversion")
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--train-bars", type=int)
    parser.add_argument("--purge-bars", type=int)
    parser.add_argument("--validate-bars", type=int)
    parser.add_argument("--embargo-bars", type=int)
    parser.add_argument("--test-bars", type=int)
    parser.add_argument("--refit-stride-bars", type=int)
    parser.add_argument("--posterior-threshold", type=float, default=0.7)
    parser.add_argument("--min-hold-bars", type=int, default=6)
    parser.add_argument("--cooldown-bars", type=int, default=4)
    parser.add_argument("--required-confirmations", type=int, default=2)
    parser.add_argument("--confidence-gap", type=float, default=0.06)
    parser.add_argument("--allow-short", action="store_true", help="Allow validated bearish regimes to map to short trades.")
    parser.add_argument("--require-daily-confirmation", action="store_true", help="Only execute 4H exposure when the daily lane agrees.")
    parser.add_argument("--require-consensus-confirmation", action="store_true", help="Only execute exposure when nearby seeds and state counts agree.")
    parser.add_argument("--consensus-gate-mode", choices=["hard", "entry_only"], default="entry_only", help="How weak consensus should be handled when the consensus filter is enabled.")
    parser.add_argument("--consensus-min-share", type=float, default=0.67, help="Minimum consensus agreement share required before a trade is allowed.")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--spread-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--impact-bps", type=float, default=2.0)
    parser.add_argument("--strict-windows", action="store_true", help="Fail instead of auto-sizing walk-forward windows.")


def _load_result(args: argparse.Namespace):
    data_config = DataConfig(symbol=args.symbol, interval=args.interval, limit=args.limit, provider=args.provider)
    model_config = ModelConfig(n_states=args.states)
    feature_columns = get_feature_columns(args.feature_pack)
    walk_config = _resolve_cli_walk_config(args)
    strategy_config = StrategyConfig(
        posterior_threshold=args.posterior_threshold,
        min_hold_bars=args.min_hold_bars,
        cooldown_bars=args.cooldown_bars,
        required_confirmations=args.required_confirmations,
        confidence_gap=args.confidence_gap,
        allow_short=args.allow_short,
        require_daily_confirmation=args.require_daily_confirmation,
        require_consensus_confirmation=args.require_consensus_confirmation,
        consensus_min_share=args.consensus_min_share,
        consensus_gate_mode=args.consensus_gate_mode,
        cost_bps=args.cost_bps,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        impact_bps=args.impact_bps,
    )

    fetched = fetch_price_data(data_config)
    feature_frame = build_feature_frame(fetched.frame, feature_columns=feature_columns)
    effective_walk_config = (
        walk_config
        if args.strict_windows
        else suggest_walk_forward_config(len(feature_frame), walk_config)[0]
    )
    result = run_walk_forward(
        feature_frame=feature_frame,
        feature_columns=feature_columns,
        interval=data_config.interval,
        model_config=model_config,
        walk_config=effective_walk_config,
        strategy_config=replace(strategy_config, require_daily_confirmation=False),
    )
    if data_config.interval == "4hour" and strategy_config.require_daily_confirmation:
        confirmation_config = DataConfig(symbol=args.symbol, interval="1day", limit=args.limit, provider=args.provider)
        confirmation_fetched = fetch_price_data(confirmation_config)
        confirmation_features = build_feature_frame(confirmation_fetched.frame, feature_columns=feature_columns)
        confirmation_walk_config = (
            default_walk_forward_config("1day")
            if args.strict_windows
            else suggest_walk_forward_config(len(confirmation_features), default_walk_forward_config("1day"))[0]
        )
        confirmation_result = run_walk_forward(
            feature_frame=confirmation_features,
            feature_columns=feature_columns,
            interval="1day",
            model_config=model_config,
            walk_config=confirmation_walk_config,
            strategy_config=replace(strategy_config, require_daily_confirmation=False),
        )
        result = apply_higher_timeframe_confirmation(
            result,
            confirmation_result,
            interval=data_config.interval,
            strategy_config=strategy_config,
            confirmation_interval="1day",
        )
    if strategy_config.require_consensus_confirmation:
        consensus = run_consensus_diagnostics(
            symbol=data_config.symbol,
            interval=data_config.interval,
            limit=data_config.limit,
            history_provider=args.provider,
            feature_columns=feature_columns,
            model_config=model_config,
            strategy_config=replace(strategy_config, require_consensus_confirmation=False),
            auto_adjust_windows=not args.strict_windows,
        )
        result = apply_consensus_confirmation(
            result,
            consensus,
            interval=data_config.interval,
            strategy_config=strategy_config,
        )
    return data_config, model_config, feature_columns, strategy_config, effective_walk_config, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Markov regime research tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest_parser = subparsers.add_parser("backtest", help="Run the walk-forward backtest")
    _common_parser(backtest_parser)

    sweep_parser = subparsers.add_parser("sweep", help="Run the parameter sweep and print top combinations")
    _common_parser(sweep_parser)

    export_parser = subparsers.add_parser("export-report", help="Export the signal report as CSV and JSON")
    _common_parser(export_parser)

    timeframe_parser = subparsers.add_parser("compare-timeframes", help="Compare 4H, 1D, and 1H using the same model controls")
    _common_parser(timeframe_parser)

    feature_parser = subparsers.add_parser("compare-feature-packs", help="Compare feature packs on the same symbol/timeframe")
    _common_parser(feature_parser)

    consensus_parser = subparsers.add_parser("consensus", help="Run nearby-state and multi-seed consensus diagnostics")
    _common_parser(consensus_parser)

    candidate_parser = subparsers.add_parser(
        "candidate-search",
        help="Rank feature pack, state count, shorting mode, and confirmation mode on deeper history",
    )
    candidate_parser.add_argument("--symbol", default="BTCUSD")
    candidate_parser.add_argument("--interval", choices=["4hour", "1day", "1hour"], default=DEFAULT_CLI_INTERVAL)
    candidate_parser.add_argument("--provider", choices=["auto", "fmp", "coinbase", "yahoo"], default="auto")
    candidate_parser.add_argument("--limit", type=int, default=5000)
    candidate_parser.add_argument("--feature-packs", default="mean_reversion,trend,baseline,regime_mix,atr_causal,trend_context")
    candidate_parser.add_argument("--state-counts", default="5,6,7,8,9")
    candidate_parser.add_argument("--short-modes", default="off,on")
    candidate_parser.add_argument("--confirmation-modes", default="off,daily,consensus_entry,daily_consensus_entry")
    candidate_parser.add_argument("--max-candidates", type=int, default=32)
    candidate_parser.add_argument("--robustness-top-k", type=int, default=2)
    candidate_parser.add_argument("--robustness-symbols", default="BTCUSD,ETHUSD,SOLUSD")
    candidate_parser.add_argument("--posterior-threshold", type=float, default=0.7)
    candidate_parser.add_argument("--min-hold-bars", type=int, default=6)
    candidate_parser.add_argument("--cooldown-bars", type=int, default=4)
    candidate_parser.add_argument("--required-confirmations", type=int, default=2)
    candidate_parser.add_argument("--confidence-gap", type=float, default=0.06)
    candidate_parser.add_argument("--cost-bps", type=float, default=10.0)
    candidate_parser.add_argument("--spread-bps", type=float, default=4.0)
    candidate_parser.add_argument("--slippage-bps", type=float, default=3.0)
    candidate_parser.add_argument("--impact-bps", type=float, default=2.0)
    candidate_parser.add_argument("--strict-windows", action="store_true")

    readiness_parser = subparsers.add_parser("readiness-audit", help="Run operational and strategy primetime readiness checks")
    _common_parser(readiness_parser)
    readiness_parser.add_argument("--robustness-symbols", default="BTCUSD,ETHUSD,SOLUSD")
    readiness_parser.add_argument("--output-dir", default="artifacts/primetime")
    readiness_parser.add_argument("--freshness-threshold-seconds", type=float, default=120.0)

    init_research_parser = subparsers.add_parser("init-research", help="Create a local research program and results TSV")
    init_research_parser.add_argument("--program", default="research_program.md")
    init_research_parser.add_argument("--results", default="results.tsv")

    autoresearch_parser = subparsers.add_parser("autoresearch", help="Run the constrained local autoresearch batch")
    autoresearch_parser.add_argument("--program", default="research_program.md")
    autoresearch_parser.add_argument("--results", default="results.tsv")

    args = parser.parse_args()

    if args.command == "init-research":
        program_path = write_research_program(args.program)
        results_path = ensure_results_tsv(args.results)
        print(f"Program: {program_path}")
        print(f"Results: {results_path}")
        return

    if args.command == "autoresearch":
        program = load_research_program(args.program)
        leaderboard = run_autoresearch(program=program, results_path=args.results)
        print(leaderboard.head(10).to_string(index=False))
        return

    if args.command == "candidate-search":
        search_strategy = StrategyConfig(
            posterior_threshold=args.posterior_threshold,
            min_hold_bars=args.min_hold_bars,
            cooldown_bars=args.cooldown_bars,
            required_confirmations=args.required_confirmations,
            confidence_gap=args.confidence_gap,
            cost_bps=args.cost_bps,
            spread_bps=args.spread_bps,
            slippage_bps=args.slippage_bps,
            impact_bps=args.impact_bps,
        )
        leaderboard = run_candidate_search(
            symbol=args.symbol,
            interval=args.interval,
            limit=args.limit,
            history_provider=args.provider,
            base_model_config=ModelConfig(),
            base_strategy_config=search_strategy,
            feature_packs=_parse_csv_strings(args.feature_packs),
            state_counts=_parse_csv_ints(args.state_counts),
            short_modes=_parse_short_modes(args.short_modes),
            confirmation_modes=_parse_csv_strings(args.confirmation_modes),
            robustness_symbols=tuple(parse_symbol_list(args.robustness_symbols)),
            auto_adjust_windows=not args.strict_windows,
            max_candidates=args.max_candidates,
            robustness_top_k=args.robustness_top_k,
        )
        summary = summarize_candidate_search(leaderboard)
        print(summary["headline"])
        print(summary["summary"])
        print()
        print(leaderboard.head(15).to_string(index=False))
        return

    if args.command == "readiness-audit":
        audit_strategy = replace(
            DEFAULT_AUDIT_STRATEGY,
            posterior_threshold=args.posterior_threshold,
            min_hold_bars=args.min_hold_bars,
            cooldown_bars=args.cooldown_bars,
            required_confirmations=args.required_confirmations,
            confidence_gap=args.confidence_gap,
            allow_short=args.allow_short,
            require_daily_confirmation=args.require_daily_confirmation,
            require_consensus_confirmation=args.require_consensus_confirmation,
            consensus_min_share=args.consensus_min_share,
            consensus_gate_mode=args.consensus_gate_mode,
            cost_bps=args.cost_bps,
            spread_bps=args.spread_bps,
            slippage_bps=args.slippage_bps,
            impact_bps=args.impact_bps,
        )
        audit = run_primetime_audit(
            repo_root=".",
            symbol=args.symbol,
            interval=args.interval,
            feature_pack=args.feature_pack,
            states=args.states,
            limit=args.limit,
            history_provider=args.provider,
            strategy_config=audit_strategy,
            walk_config=_resolve_cli_walk_config(args),
            strict_windows=args.strict_windows,
            robustness_symbols=tuple(parse_symbol_list(args.robustness_symbols)),
            freshness_threshold_seconds=args.freshness_threshold_seconds,
        )
        report_dir = write_primetime_audit_report(audit, output_dir=args.output_dir)
        print(f"Report verdict: {audit.report_summary['verdict']}")
        print(audit.report_summary["summary"])
        print("\nPlatform")
        print(audit.platform_gates.to_string(index=False))
        print("\nStrategy")
        print(audit.strategy_gates.to_string(index=False))
        print(f"\nCurrent action: {audit.action_plan['action']}")
        print(audit.action_plan["entry_guide"])
        print(f"\nReport directory: {report_dir}")
        return

    data_config, model_config, feature_columns, strategy_config, walk_config, result = _load_result(args)

    if args.command == "backtest":
        for key, value in result.metrics.items():
            print(f"{key}: {value:.6f}")
        return

    if args.command == "sweep":
        sweep_results = parameter_sweep(
            predictions=result.predictions,
            n_states=args.states,
            base_config=strategy_config,
            sweep_config=SweepConfig(),
            interval=data_config.interval,
        )
        print(sweep_results.head(10).to_string(index=False))
        return

    if args.command == "compare-timeframes":
        timeframe_results = run_timeframe_comparison(
            symbol=data_config.symbol,
            limit=data_config.limit,
            model_config=model_config,
            strategy_config=strategy_config,
            feature_pack=args.feature_pack,
            feature_columns=feature_columns,
            history_provider=args.provider,
            auto_adjust_windows=not args.strict_windows,
        )
        print(timeframe_results.to_string(index=False))
        return

    if args.command == "compare-feature-packs":
        fetched = fetch_price_data(data_config)
        feature_results = run_feature_pack_comparison(
            price_frame=fetched.frame,
            interval=data_config.interval,
            model_config=model_config,
            strategy_config=strategy_config,
            symbol=data_config.symbol,
            limit=data_config.limit,
            history_provider=args.provider,
            auto_adjust_windows=not args.strict_windows,
        )
        print(feature_results.to_string(index=False))
        return

    if args.command == "consensus":
        consensus = run_consensus_diagnostics(
            symbol=data_config.symbol,
            interval=data_config.interval,
            limit=data_config.limit,
            history_provider=args.provider,
            feature_columns=feature_columns,
            model_config=model_config,
            strategy_config=strategy_config,
            auto_adjust_windows=not args.strict_windows,
        )
        print("Summary")
        print(consensus.summary.to_string(index=False))
        print("\nMembers")
        print(consensus.members.to_string(index=False))
        return

    exported = export_signal_report(result.predictions, symbol=data_config.symbol, interval=data_config.interval)
    print(f"CSV: {exported['csv']}")
    print(f"JSON: {exported['json']}")
