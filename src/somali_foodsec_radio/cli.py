"""Command-line entry point for the ``radio-collect`` console script.

``radio-collect`` downloads Radio Ergo broadcasts from SoundCloud and transcribes them
with the Somali Wav2Vec2 ASR model. Heavy dependencies are imported lazily so that
``radio-collect --help`` works even without the ``[asr]`` extras installed.
"""

from __future__ import annotations

import argparse


def _config() -> dict:
    """Load project config, falling back to an empty dict so ``--help`` always works."""
    try:
        from .config import get_config

        return get_config()
    except Exception:  # noqa: BLE001 - never let config issues block --help
        return {}


def build_parser() -> argparse.ArgumentParser:
    """Build the ``radio-collect`` argument parser (defaults sourced from config)."""
    cfg = _config()
    soundcloud = cfg.get("soundcloud", {})
    asr = cfg.get("asr", {})
    paths = cfg.get("paths", {})

    parser = argparse.ArgumentParser(
        prog="radio-collect",
        description=(
            "Download Radio Ergo broadcasts from SoundCloud and transcribe them with "
            "the Somali Wav2Vec2 ASR model."
        ),
    )
    parser.add_argument("--start", required=True, help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date, YYYY-MM-DD")
    parser.add_argument(
        "--profile",
        default=soundcloud.get("profile_url", "https://soundcloud.com/radio-ergo"),
        help="SoundCloud profile URL",
    )
    parser.add_argument(
        "--output",
        default=str(paths.get("transcripts", "data/interim/transcripts")),
        help="Output directory for transcripts and the CSV database",
    )
    parser.add_argument(
        "--batch-days",
        type=int,
        default=asr.get("batch_size_days", 30),
        help="Process the date range in chunks of N days (crash recovery)",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=asr.get("batch_size", 8),
        help="GPU inference batch size (4-16 recommended)",
    )
    parser.add_argument(
        "--no-fp16", action="store_true", help="Disable FP16 mixed precision"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the GPU-optimised collection pipeline."""
    args = build_parser().parse_args(argv)

    # Lazy import: keeps `radio-collect --help` working without the [asr] extras.
    from .collection.streaming import run_batch_collection

    run_batch_collection(
        profile_url=args.profile,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output,
        batch_size_days=args.batch_days,
        gpu_batch_size=args.gpu_batch_size,
        use_fp16=not args.no_fp16,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
