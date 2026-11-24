"""Tests for the one-writer-per-archive sync lock."""

from __future__ import annotations

from pathlib import Path

import pytest

from granola_exporter.store import Archive, ArchiveBusy


@pytest.fixture
def isolated_state(tmp_path: Path, monkeypatch) -> Path:
    """Keep the lock out of the developer's real state directory."""
    state = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    return state


def test_exclusive_refuses_a_second_writer(tmp_path: Path, isolated_state: Path):
    """A second sync of the same archive must be turned away, not run."""
    archive = Archive(tmp_path / "archive")
    other = Archive(tmp_path / "archive")

    with archive.exclusive():
        with pytest.raises(ArchiveBusy):
            with other.exclusive():
                pass


def test_exclusive_does_not_create_the_archive(tmp_path: Path, isolated_state: Path):
    """Taking the lock must not bring an archive directory into existence.

    A sync that never starts, because the user is logged out, has to leave
    nothing behind.
    """
    root = tmp_path / "archive"
    archive = Archive(root)

    with archive.exclusive():
        pass

    assert not root.exists()


def test_exclusive_separates_different_archives(tmp_path: Path, isolated_state: Path):
    """Two archives must not contend with each other."""
    one = Archive(tmp_path / "one")
    two = Archive(tmp_path / "two")

    with one.exclusive():
        with two.exclusive():
            pass


def test_exclusive_releases_on_exit(tmp_path: Path, isolated_state: Path):
    """The next sync can take the archive once the first has finished."""
    archive = Archive(tmp_path / "archive")

    with archive.exclusive():
        pass
    with archive.exclusive():
        pass


def test_exclusive_releases_when_the_body_raises(tmp_path: Path, isolated_state: Path):
    """A crashed sync must not leave the archive locked."""
    archive = Archive(tmp_path / "archive")

    with pytest.raises(RuntimeError, match="boom"):
        with archive.exclusive():
            raise RuntimeError("boom")

    with archive.exclusive():
        pass
