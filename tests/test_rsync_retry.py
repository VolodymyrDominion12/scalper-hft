"""Retry wrapper for live parquet rsync (exit 23/24)."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "rsync_retry.sh"


def _fake_rsync(bin_dir: Path, body: str) -> None:
    rsync = bin_dir / "rsync"
    rsync.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    rsync.chmod(rsync.stat().st_mode | stat.S_IEXEC)


def _run(tmp_path: Path, *args: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["RSYNC_RETRY_SLEEP"] = "0"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(_SCRIPT), *args],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_rsync_retry_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(_SCRIPT)], check=True)


def test_rsync_retry_requires_args() -> None:
    r = subprocess.run(["bash", str(_SCRIPT)], cwd=_REPO, capture_output=True, text=True, check=False)
    assert r.returncode == 2
    assert "usage:" in r.stderr


def test_rsync_retry_succeeds_first_try(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    counter = tmp_path / "n"
    counter.write_text("0", encoding="utf-8")
    _fake_rsync(
        bin_dir,
        f'n=$(cat "{counter}"); echo $((n+1)) > "{counter}"; echo rsync-ok "$@"; exit 0',
    )
    r = _run(tmp_path, "-avz", "src/", "dst/")
    assert r.returncode == 0, r.stderr + r.stdout
    assert counter.read_text(encoding="utf-8").strip() == "1"


def test_rsync_retry_recovers_from_23(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    counter = tmp_path / "n"
    counter.write_text("0", encoding="utf-8")
    _fake_rsync(
        bin_dir,
        f"""
n=$(cat "{counter}")
n=$((n+1))
echo "$n" > "{counter}"
echo "rsync-stub $@"
if [ "$n" -eq 1 ]; then exit 23; fi
exit 0
""",
    )
    r = _run(tmp_path, "-avzP", "--whole-file", "src/", "dst/")
    assert r.returncode == 0, r.stderr + r.stdout
    assert "повтор" in r.stderr
    assert counter.read_text(encoding="utf-8").strip() == "2"


def test_rsync_retry_gives_up_after_max_23(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_rsync(bin_dir, "exit 23")
    r = _run(tmp_path, "src/", "dst/", extra_env={"RSYNC_RETRIES": "2"})
    assert r.returncode == 23
    assert "після 2 спроб" in r.stderr


def test_rsync_retry_does_not_retry_hard_errors(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    counter = tmp_path / "n"
    counter.write_text("0", encoding="utf-8")
    _fake_rsync(
        bin_dir,
        f'n=$(cat "{counter}"); echo $((n+1)) > "{counter}"; exit 1',
    )
    r = _run(tmp_path, "src/", "dst/")
    assert r.returncode == 1
    assert counter.read_text(encoding="utf-8").strip() == "1"
