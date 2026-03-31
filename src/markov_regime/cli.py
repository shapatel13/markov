from __future__ import annotations

import argparse

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
from markov_regime.research import (
    ensure_results_tsv,
    load_research_program,
    run_autoresearch,
    run_feature_pack_comparison,
    run_timeframe_comparison,
    write_research_program,
)
from markov_regime.reporting import export_signal_report
from markov_regime.strategy import parameter_sweep
from markov_regime.walkforward import run_walk_forward, suggest_walk_forward_config

DEFAULT_CLI_INTERVAL = "4hour"


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
    parser.add_argument("--feature-pack", choices=list(list_feature_packs()), default="baseline")
    parser.add_argument("--states", type=int, default=6)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--train-bars", type=int)
    parser.add_argument("--purge-bars", type=int)
    parser.add_argument("--validate-bars", type=int)
    parser.add_argument("--embargo-bars", type=int)
    parser.add_argument("--test-bars", type=int)
    parser.add_argument("--refit-stride-bars", type=int)
    parser.add_argument("--posterior-threshold", type=float, default=0.65)
    parser.add_argument("--min-hold-bars", type=int, default=6)
    parser.add_argument("--cooldown-bars", type=int, default=3)
    parser.add_argument("--required-confirmations", type=int, default=2)
    parser.add_argument("--confidence-gap", type=float, default=0.05)
    parser.add_argument("--cost-bps", type=float, default=2.0)
    parser.add_argument("--spread-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--impact-bps", type=float, default=2.0)
    parser.add_argument("--strict-windows", action="store_true", help="Fail instead of auto-sizing walk-forward windows.")


def _load_result(args: argparse.Namespace):
    data_config = DataConfig(symbol=args.symbol, interval=args.interval, limit=args.limit)
    model_config = ModelConfig(n_states=args.states)
    feature_columns = get_feature_columns(args.feature_pack)
    walk_config = _resolve_cli_walk_config(args)
    strategy_config = StrategyConfig(
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
            auto_adjust_windows=not args.strict_windows,
        )
        print(feature_results.to_string(index=False))
        return

    exported = export_signal_report(result.predictions, symbol=data_config.symbol, interval=data_config.interval)
    print(f"CSV: {exported['csv']}")
    print(f"JSON: {exported['json']}")
