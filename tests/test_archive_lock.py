"""Tests for the one-writer-per-archive sync lock."""

from __future__ import annotations

import errno
import fcntl
import os
import stat
from pathlib import Path

import pytest

from granola_exporter.secure_io import FILE_MODE
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


@pytest.fixture
def permissive_umask():
    """Run under the umask most machines have, so modes are not luck.

    Yields:
        Nothing; the previous umask is restored afterwards.
    """
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


def test_lock_is_owner_only(tmp_path: Path, isolated_state: Path, permissive_umask):
    """A fresh lock file is owner-only even under a permissive umask."""
    archive = Archive(tmp_path / "archive")

    with archive.exclusive():
        assert stat.S_IMODE(archive.lock_path.stat().st_mode) == FILE_MODE


def test_a_loose_lock_file_is_tightened(
    tmp_path: Path, isolated_state: Path, permissive_umask
):
    """An existing lock left world-writable is tightened on the next sync.

    ``O_CREAT`` does not touch the mode of a file that already exists, and a
    writable lock on a shared machine lets another account hold it. This
    matches ``secure_mkdir``, which also tightens what it finds.
    """
    archive = Archive(tmp_path / "archive")
    archive.lock_path.parent.mkdir(parents=True, exist_ok=True)
    archive.lock_path.touch()
    archive.lock_path.chmod(0o666)

    with archive.exclusive():
        pass

    assert stat.S_IMODE(archive.lock_path.stat().st_mode) == FILE_MODE


def test_one_archive_spelled_two_ways_takes_one_lock(
    tmp_path: Path, isolated_state: Path
):
    """Case differences must not hand the same archive two locks.

    Only meaningful on a case-insensitive filesystem, which is the macOS
    default and where this bug would actually bite.
    """
    root = tmp_path / "Archive"
    root.mkdir()
    upper = Archive(root)
    lower = Archive(Path(str(root).replace("Archive", "archive")))

    if not os.path.samefile(upper.root, lower.root):
        pytest.skip("case-sensitive filesystem")

    assert upper.lock_path == lower.lock_path


def test_a_filesystem_that_cannot_lock_is_not_reported_as_busy(
    tmp_path: Path, isolated_state: Path, monkeypatch
):
    """ENOLCK and friends must propagate, not masquerade as a second sync."""

    def refuse(fd, operation):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", refuse)
    archive = Archive(tmp_path / "archive")

    with pytest.raises(OSError) as caught:
        with archive.exclusive():
            pass
    assert not isinstance(caught.value, ArchiveBusy)
    assert caught.value.errno == errno.ENOLCK


def test_interrupt_saves_the_index_while_the_lock_is_still_held(
    tmp_path: Path, isolated_state: Path, monkeypatch, capsys
):
    """Ctrl-C writes the index, so that write must happen inside the lock.

    Regression test: the interrupt handler used to sit outside the ``with``
    block, so unwinding the context manager released the lock before
    ``save_index`` ran -- the one write on the interrupt path was the one
    write a concurrent sync could still clobber.
    """
    from granola_exporter import cli

    root = tmp_path / "archive"
    root.mkdir()
    monkeypatch.setenv("GRANOLA_ARCHIVE_DIR", str(root))
    monkeypatch.setenv("GRANOLA_SYNC_SOURCE", "mcp")

    held_during_save: list[bool] = []
    lock_path = Archive(root).lock_path

    def probe_then_save(self) -> None:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held_during_save.append(False)
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                held_during_save.append(True)
        finally:
            os.close(fd)

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(Archive, "save_index", probe_then_save)
    monkeypatch.setattr(cli, "_sync_via_mcp", interrupt)

    assert cli.main(["sync", "--source", "mcp"]) == 130
    assert held_during_save == [True], "index was saved after the lock released"
