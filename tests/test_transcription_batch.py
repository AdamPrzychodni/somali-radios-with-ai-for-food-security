"""Tests for the generic directory batch runner."""

from somali_foodsec_radio.transcription.batch import run_directory_batch


def test_processes_all_matching_files(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    out_dir = tmp_path / "out"
    (in_dir / "a.txt").write_text("a", encoding="utf-8")
    (in_dir / "b.txt").write_text("b", encoding="utf-8")

    results = run_directory_batch(
        str(in_dir), str(out_dir), work_fn=lambda p: "done", input_glob="*.txt"
    )

    assert results == {"a.txt": "success", "b.txt": "success"}
    assert (out_dir / "a.txt").read_text(encoding="utf-8") == "done"
    assert (out_dir / "b.txt").read_text(encoding="utf-8") == "done"


def test_skips_existing_outputs(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (in_dir / "a.txt").write_text("a", encoding="utf-8")
    (out_dir / "a.txt").write_text("already there", encoding="utf-8")

    results = run_directory_batch(
        str(in_dir), str(out_dir), work_fn=lambda p: "new", input_glob="*.txt"
    )

    assert results["a.txt"] == "skipped"
    assert (out_dir / "a.txt").read_text(encoding="utf-8") == "already there"


def test_records_failures_and_writes_log(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    out_dir = tmp_path / "out"
    (in_dir / "a.txt").write_text("a", encoding="utf-8")

    def boom(path):
        raise RuntimeError("kaboom")

    results = run_directory_batch(
        str(in_dir),
        str(out_dir),
        work_fn=boom,
        input_glob="*.txt",
        retry_count=2,
        delay_between_failures=0,
        failure_log="failed.json",
    )

    assert results["a.txt"].startswith("failed:")
    assert "kaboom" in (out_dir / "failed.json").read_text(encoding="utf-8")
