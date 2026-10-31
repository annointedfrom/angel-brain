"""Tests for the public-release privacy gate (make_public_release.py)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make_public_release import (  # noqa: E402
    DATA_FILES,
    release_files,
    scan_file,
    scan_text,
)


def test_detects_email():
    assert "email" in scan_text("contact me at someone@example.com")


def test_detects_windows_user_path():
    assert "windows user path" in scan_text(r"see C:\Users\someone\notes.md")


def test_detects_posix_home_path():
    assert "posix home path" in scan_text("cd /home/someone/training")


def test_detects_api_key_shape():
    assert "api key" in scan_text("key = sk-abcdefghijklmnopqrstuv")


def test_clean_text_passes():
    assert scan_text("Default to the NAB. Spell out First Corinthians.") == []


def test_raw_vault_dump_is_never_released():
    assert "vault_persona.jsonl" not in release_files(include_data=True)


def test_data_excluded_by_default():
    assert not set(DATA_FILES) & set(release_files(include_data=False))


def test_allowlisted_files_pass_scan():
    root = Path(__file__).resolve().parents[1]
    hits = [h for f in release_files(include_data=True) if (root / f).is_file() for h in scan_file(f)]
    assert hits == [], hits
