"""Command-line interface for training and checkpoint inference."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any, Sequence

from . import (
    ScRAWConfig,
    __version__,
    load_config,
    run_inference_from_checkpoint,
    run_pipeline,
)
from .diagnostics import inspect_reference_environment


def _json_safe(value: Any) -> Any:
    """Recursively convert scientific values to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None

    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())

    return value


def _print_json(payload: Any) -> None:
    """Write stable, human-readable JSON to standard output."""
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False))


def _load_training_config(path: str | None) -> ScRAWConfig:
    """Load a user config or return the built-in defaults."""
    return ScRAWConfig() if path is None else load_config(path)


def _ensure_output_available(path: str, overwrite: bool) -> None:
    """Refuse accidental writes into an existing non-empty directory."""
    output_path = Path(path).expanduser()
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"output path is not a directory: {output_path}")
    if output_path.is_dir() and any(output_path.iterdir()) and not overwrite:
        raise ValueError(
            f"output directory is not empty: {output_path}; "
            "pass --overwrite to authorize replacing pipeline-managed files"
        )


def _configure_logging(verbosity: int) -> None:
    """Map CLI verbosity to the standard logging levels."""
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _apply_run_overrides(config: ScRAWConfig, args: argparse.Namespace) -> None:
    """Apply explicit command-line values without replacing config defaults."""
    if args.data_path is not None:
        config.data.data_path = args.data_path
    if args.output_dir is not None:
        config.data.output_dir = args.output_dir
    if args.label_key is not None:
        config.data.label_key = args.label_key
    if args.batch_key is not None:
        config.batch_correction.key = args.batch_key
    if args.input_mode is not None:
        config.preprocessing.input_mode = args.input_mode
    if args.device is not None:
        config.runtime.device = args.device
    if args.seed is not None:
        config.runtime.seed = args.seed
    if args.no_figures:
        config.outputs.save_figures = False
    if args.no_model:
        config.outputs.save_model = False


def _result_summary(mode: str, result: dict[str, Any]) -> dict[str, Any]:
    """Keep terminal output concise while retaining the main run result."""
    summary = {
        "mode": mode,
        "output_dir": result.get("output_dir"),
        "label_key": result.get("label_key"),
        "batch_key": result.get("batch_key"),
        "metrics": result.get("metrics", {}),
    }
    checkpoint_path = result.get("checkpoint_path")
    if checkpoint_path is not None:
        summary["checkpoint_path"] = checkpoint_path
    for field in ("known_label_count", "effective_pseudo_k"):
        if field in result:
            summary[field] = result[field]
    return summary


def _run_command(args: argparse.Namespace) -> int:
    """Execute a full training run."""
    config = _load_training_config(args.config)
    _apply_run_overrides(config, args)
    _ensure_output_available(config.data.output_dir, overwrite=args.overwrite)
    result = run_pipeline(config)
    _print_json(_result_summary("training", result))
    return 0


def _infer_command(args: argparse.Namespace) -> int:
    """Replay preprocessing and clustering from a saved checkpoint."""
    _ensure_output_available(args.output_dir, overwrite=args.overwrite)
    result = run_inference_from_checkpoint(
        config=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        data_path=args.data_path,
        device=args.device,
    )
    _print_json(_result_summary("inference_only", result))
    return 0


def _show_config_command(args: argparse.Namespace) -> int:
    """Print the built-in defaults or a normalized config file."""
    config = _load_training_config(args.config)
    _print_json(config.to_dict())
    return 0


def _doctor_command(args: argparse.Namespace) -> int:
    """Report compatibility with the exact reference environment."""
    report = inspect_reference_environment(args.repository_root)
    _print_json(report)
    if args.reference and not report["reference_compatible"]:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser without importing scientific dependencies."""
    parser = argparse.ArgumentParser(
        prog="scraw",
        description="Train scRAW or run inference from a saved checkpoint.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"scraw {__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging detail; repeat for debug logging.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show a traceback instead of a concise error when a run fails.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Train scRAW and write clustering outputs.",
        description="Train scRAW from built-in defaults or a JSON configuration.",
    )
    run_parser.add_argument(
        "--config",
        help="JSON config path. Omit it to use the built-in defaults.",
    )
    run_parser.add_argument("--data-path", help="Override the input .h5ad path.")
    run_parser.add_argument("--output-dir", help="Override the output directory.")
    run_parser.add_argument("--label-key", help="Override the evaluation label column.")
    run_parser.add_argument("--batch-key", help="Override the batch column.")
    run_parser.add_argument(
        "--input-mode",
        choices=("auto", "raw", "preprocessed"),
        help="Override input handling; auto preserves the default detection behavior.",
    )
    run_parser.add_argument(
        "--device",
        help="Override the runtime device, for example auto, cpu, cuda, or mps.",
    )
    run_parser.add_argument("--seed", type=int, help="Override the random seed.")
    run_parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Disable figure generation for this run.",
    )
    run_parser.add_argument(
        "--no-model",
        action="store_true",
        help="Do not save the trained autoencoder checkpoint.",
    )
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of pipeline-managed files in a non-empty output directory.",
    )
    run_parser.set_defaults(handler=_run_command)

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run inference and clustering from a saved checkpoint.",
        description="Replay preprocessing, encoding, and clustering from saved weights.",
    )
    infer_parser.add_argument(
        "--config",
        required=True,
        help="Training config JSON, normally config/config_used.json.",
    )
    infer_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Saved autoencoder checkpoint path.",
    )
    infer_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for inference-only outputs.",
    )
    infer_parser.add_argument("--data-path", help="Override the input .h5ad path.")
    infer_parser.add_argument(
        "--device",
        help="Override the runtime device, for example auto, cpu, cuda, or mps.",
    )
    infer_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of pipeline-managed files in a non-empty output directory.",
    )
    infer_parser.set_defaults(handler=_infer_command)

    config_parser = subparsers.add_parser(
        "show-config",
        help="Print a normalized configuration as JSON.",
    )
    config_parser.add_argument(
        "--config",
        help="Config to normalize. Omit it to print the built-in defaults.",
    )
    config_parser.set_defaults(handler=_show_config_command)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Inspect compatibility with the exact reference environment.",
    )
    doctor_parser.add_argument(
        "--reference",
        action="store_true",
        help="Exit with status 1 when any exact reference check fails.",
    )
    doctor_parser.add_argument(
        "--repository-root",
        help="Source checkout containing requirements.txt, configs/, and data/.",
    )
    doctor_parser.set_defaults(handler=_doctor_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and convert expected user-facing failures to exit code 1."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("scraw: interrupted", file=sys.stderr)
        return 130
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        if args.debug:
            raise
        print(f"scraw: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
