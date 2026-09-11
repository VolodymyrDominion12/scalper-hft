"""Перевірка профілів .env: жоден приклад не ставить DRY_RUN=false."""

from __future__ import annotations

from pathlib import Path


def _parse_env_bool(raw: str) -> bool | None:
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        if key.strip() == "DRY_RUN":
            v = val.strip().lower()
            if v in {"1", "true", "yes", "on"}:
                return True
            if v in {"0", "false", "no", "off"}:
                return False
    return None


def test_env_profile_examples_dry_run_true() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in ("docs/env/research.env.example", "docs/env/vps-paper.env.example"):
        text = (root / rel).read_text(encoding="utf-8")
        assert _parse_env_bool(text) is True, f"{rel} must set DRY_RUN=true"
