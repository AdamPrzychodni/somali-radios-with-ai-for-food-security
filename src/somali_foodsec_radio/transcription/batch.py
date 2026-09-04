"""Generic directory batch runner with skip-existing, retry and failure logging.

Replaces the near-identical ``process_all_mp3_files`` / ``process_all_transcription_files``
loops from notebooks/05: the engine-specific work is injected as a *work_fn* callable.
"""

from __future__ import annotations

import glob
import json
import os
import time
from collections.abc import Callable
from datetime import datetime

from ..config import get_setting


def run_directory_batch(
    input_dir: str,
    output_dir: str,
    work_fn: Callable[[str], str],
    input_glob: str = "*.mp3",
    output_suffix: str = ".txt",
    retry_count: int | None = None,
    delay_between_failures: int | None = None,
    failure_log: str = "failed.json",
) -> dict[str, str]:
    """Run *work_fn* over every file in *input_dir* matching *input_glob*.

    For each input file, ``<output_dir>/<stem><output_suffix>`` is written with the
    string returned by ``work_fn(input_path)``. Files whose output already exists are
    skipped; failures are retried and, if still failing, appended to *failure_log*.

    Returns:
        A dict mapping each input filename to its status: ``"success"``, ``"skipped"``,
        or ``"failed: <error>"``.

    Retry behaviour defaults to ``retry.count`` / ``retry.delay_seconds`` in the config.
    """
    if retry_count is None:
        retry_count = get_setting("retry.count", 3)
    if delay_between_failures is None:
        delay_between_failures = get_setting("retry.delay_seconds", 10)

    os.makedirs(output_dir, exist_ok=True)
    source_files = glob.glob(os.path.join(input_dir, input_glob))

    results: dict[str, str] = {}
    for src in source_files:
        filename = os.path.basename(src)
        stem = os.path.splitext(filename)[0]
        output_file = os.path.join(output_dir, f"{stem}{output_suffix}")

        if os.path.exists(output_file):
            print(f"Skipping {filename} - already processed")
            results[filename] = "skipped"
            continue

        print(f"Processing {filename}...")
        for attempt in range(retry_count):
            try:
                result_text = work_fn(src)
                with open(output_file, "w", encoding="utf-8") as fh:
                    fh.write(result_text)
                print(f"Successfully processed {filename}")
                results[filename] = "success"
                break
            except Exception as exc:  # noqa: BLE001 - retry, then record the failure
                print(
                    f"Attempt {attempt + 1}/{retry_count} failed for {filename}: {exc}"
                )
                if attempt < retry_count - 1:
                    print(f"Waiting {delay_between_failures}s before retrying...")
                    time.sleep(delay_between_failures)
                else:
                    print(f"All attempts failed for {filename}")
                    results[filename] = f"failed: {exc}"
                    with open(
                        os.path.join(output_dir, failure_log), "a", encoding="utf-8"
                    ) as fh:
                        fh.write(
                            json.dumps(
                                {
                                    "filename": filename,
                                    "path": src,
                                    "timestamp": datetime.now().isoformat(),
                                    "error": str(exc),
                                }
                            )
                            + "\n"
                        )
    return results
