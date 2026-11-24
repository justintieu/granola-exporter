"""Tests for the owner-only filesystem primitives."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from granola_exporter.secure_io import FILE_MODE, write_json


def test_write_json_uses_a_unique_temp_name_per_call(tmp_path: Path, monkeypatch):
    """Two writers of the same destination must not share a temp path.

    The temp name used to be derived from the destination, so two syncs that
    reached the same meeting both wrote ``raw.json.tmp``. Whichever renamed
    first left the other's rename with nothing to move, crashing a run that
    had done nothing wrong.
    """
    target = tmp_path / "raw.json"
    seen: list[Path] = []
    original = Path.replace

    def record(self: Path, dest) -> Path:
        seen.append(self)
        return original(self, dest)

    monkeypatch.setattr(Path, "replace", record)

    write_json(target, {"writer": 1})
    write_json(target, {"writer": 2})

    assert len(seen) == 2
    assert seen[0] != seen[1], "both writers would race on the same temp file"


def test_write_json_leaves_an_owner_only_file_and_no_temp(tmp_path: Path):
    """The destination is owner-only and the temp file does not linger."""
    target = tmp_path / "raw.json"

    write_json(target, {"id": "x"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"id": "x"}
    assert stat.S_IMODE(target.stat().st_mode) == FILE_MODE
    assert not list(tmp_path.glob("*.tmp")), "temp file left behind"


def test_write_json_removes_its_temp_file_when_the_write_fails(tmp_path: Path):
    """A failed write must not leave a temp file for a caller that globs."""
    target = tmp_path / "raw.json"

    class Exploding:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    try:
        write_json(target, {"bad": Exploding()})
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected the write to fail")

    assert not list(tmp_path.glob("*.tmp")), "temp file left behind"


def test_write_json_does_not_leak_the_descriptor_when_fdopen_fails(
    tmp_path: Path, monkeypatch
):
    """``fdopen`` does not own the fd until it succeeds, so it must be closed.

    Counting with ``dup`` rather than ``/dev/fd``: the lowest descriptor the
    kernel hands out is stable if nothing leaked, and climbs by one per leak.
    """
    target = tmp_path / "raw.json"

    def lowest_free_fd() -> int:
        fd = os.dup(0)
        os.close(fd)
        return fd

    def explode(*_args, **_kwargs):
        raise OSError("no streams today")

    before = lowest_free_fd()
    monkeypatch.setattr(os, "fdopen", explode)

    for _ in range(10):
        with pytest.raises(OSError, match="no streams today"):
            write_json(target, {"x": 1})

    monkeypatch.undo()
    assert lowest_free_fd() == before, "write_json leaked a descriptor"
