from __future__ import annotations

import argparse

from markov_regime.config import DataConfig, ModelConfig, StrategyConfig, SweepConfig, WalkForwardConfig
from markov_regime.data import fetch_price_data
from markov_regime.features import FEATURE_COLUMNS, build_feature_frame
from markov_regime.reporting import export_signal_report
from markov_regime.strategy import parameter_sweep
from markov_regime.walkforward import run_walk_forward


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--interval", choices=["1hour", "1day"], default="1hour")
    parser.add_argument("--states", type=int, default=6)
    parser.add_argument("--limit", type=int, default=2500)
    parser.add_argument("--train-bars", type=int, default=750)
    parser.add_argument("--validate-bars", type=int, default=180)
    parser.add_argument("--test-bars", type=int, default=180)
    parser.add_argument("--refit-stride-bars", type=int, default=180)
    parser.add_argument("--posterior-threshold", type=float, default=0.65)
    parser.add_argument("--min-hold-bars", type=int, default=6)
    parser.add_argument("--cooldown-bars", type=int, default=3)
    parser.add_argument("--required-confirmations", type=int, default=2)
    parser.add_argument("--cost-bps", type=float, default=2.0)


def _load_result(args: argparse.Namespace):
    data_config = DataConfig(symbol=args.symbol, interval=args.interval, limit=args.limit)
    model_config = ModelConfig(n_states=args.states)
    walk_config = WalkForwardConfig(
        train_bars=args.train_bars,
        validate_bars=args.validate_bars,
        test_bars=args.test_bars,
        refit_stride_bars=args.refit_stride_bars,
    )
    strategy_config = StrategyConfig(
        posterior_threshold=args.posterior_threshold,
        min_hold_bars=args.min_hold_bars,
        cooldown_bars=args.cooldown_bars,
        required_confirmations=args.required_confirmations,
        cost_bps=args.cost_bps,
    )

    fetched = fetch_price_data(data_config)
    feature_frame = build_feature_frame(fetched.frame)
    result = run_walk_forward(
        feature_frame=feature_frame,
        feature_columns=FEATURE_COLUMNS,
        interval=data_config.interval,
        model_config=model_config,
        walk_config=walk_config,
        strategy_config=strategy_config,
    )
    return data_config, strategy_config, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Markov regime research tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest_parser = subparsers.add_parser("backtest", help="Run the walk-forward backtest")
    _common_parser(backtest_parser)

    sweep_parser = subparsers.add_parser("sweep", help="Run the parameter sweep and print top combinations")
    _common_parser(sweep_parser)

    export_parser = subparsers.add_parser("export-report", help="Export the signal report as CSV and JSON")
    _common_parser(export_parser)

    args = parser.parse_args()
    data_config, strategy_config, result = _load_result(args)

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

    exported = export_signal_report(result.predictions, symbol=data_config.symbol, interval=data_config.interval)
    print(f"CSV: {exported['csv']}")
    print(f"JSON: {exported['json']}")

