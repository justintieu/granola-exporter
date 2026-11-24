"""Archive layout, index and sync state.

Layout::

    archive/
      2026/01/2026-01-27--quarterly-yoghurt-budget--not_1d3tmYTlCICgjy/
        note.md
        transcript.md
        raw.json
      index.json
      .sync-state.json

``raw.json`` is written before any rendering, so changing a Markdown template
never requires refetching from the API.

The archive is append-mostly. A note that disappears upstream is *never*
deleted locally -- it is flagged ``upstream_missing`` in the index, because
surviving upstream deletion is the entire point of keeping an archive.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Iterator
from fnmatch import fnmatch
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    SOURCE_PUBLIC_API,
    Note,
    is_valid_archive_key,
)
from .secure_io import DIR_MODE, FILE_MODE
from .secure_io import read_json as _read_json
from .secure_io import secure_mkdir as _secure_mkdir
from .secure_io import secure_write_text as _secure_write_text
from .secure_io import write_json as _write_json

# Re-exported: the 0600/0700 guarantee is documented in SECURITY.md and asserted
# by tests against this module, so the names stay importable from here.
__all__ = [
    "DIR_MODE",
    "FILE_MODE",
    "Archive",
    "SyncResult",
    "UnsafeArchivePathError",
    "content_hash",
    "slugify",
]

INDEX_NAME = "index.json"
STATE_NAME = ".sync-state.json"
LOCK_NAME = "sync.lock"
STATE_SCHEMA_VERSION = 2
RAW_NAME = "raw.json"
NOTE_NAME = "note.md"
TRANSCRIPT_NAME = "transcript.md"

MAX_SLUG_LEN = 60
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")

# How many recently archived notes infer_account_email reads. Enough for one
# colleague's meeting not to outvote the owner, small enough to stay free.
ACCOUNT_SAMPLE = 25


class UnsafeArchivePathError(ValueError):
    """Raised when a computed path would fall outside the archive root.

    Note ids and index entries are attacker-influenced in the sense that they
    come from the API or from a file on disk. Building paths from them without
    a containment check would allow writes outside the archive.
    """


class ArchiveBusy(RuntimeError):
    """Raised when another process already holds the archive for writing."""

    def __init__(self, root: Path) -> None:
        """Initialize the error.

        Args:
            root: The archive another process is syncing.
        """
        super().__init__(f"another sync is already running in {root}")
        self.root = root


def _resolve_within(root: Path, *parts: str) -> Path:
    """Join path components under ``root`` and verify the result stays inside.

    Guards against both ``..`` traversal and absolute components -- note that
    ``Path("/a") / "/etc"`` yields ``/etc``, silently discarding the base.

    Args:
        root: The archive root that the result must remain inside.
        *parts: Path components to join beneath ``root``.

    Returns:
        The resolved path, guaranteed to be ``root`` or a descendant of it.

    Raises:
        UnsafeArchivePathError: If the joined path escapes ``root``.
    """
    root_resolved = root.resolve()
    candidate = root_resolved.joinpath(*parts).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise UnsafeArchivePathError(
            f"path escapes the archive root: {'/'.join(parts)!r}"
        )
    return candidate


def _owner_email(raw: dict[str, Any]) -> str:
    """Read the owning account's email out of an archived payload.

    Args:
        raw: A decoded ``raw.json``, from either backend.

    Returns:
        The email, or ``""`` when the payload records none.
    """
    owner = raw.get("owner")
    if isinstance(owner, dict) and owner.get("email"):
        return str(owner["email"]).strip()

    mcp = raw.get("mcp")
    if not isinstance(mcp, dict):
        return ""
    # Imported lazily: the public API path has no reason to pay for the MCP
    # parser, matching how the rest of the codebase defers that import.
    from .mcp_parse import creator_email

    for key in ("get_meetings_element", "list_meetings_element"):
        element = mcp.get(key)
        if isinstance(element, str) and (email := creator_email(element)):
            return email
    return ""


def slugify(title: str) -> str:
    """Reduce a meeting title to a filesystem-safe slug.

    Args:
        title: The raw meeting title.

    Returns:
        A lowercase hyphenated slug, truncated to a sane length, or
        ``"untitled"`` when the title yields nothing usable.
    """
    slug = _SLUG_STRIP.sub("-", title.lower()).strip("-")
    if len(slug) > MAX_SLUG_LEN:
        slug = slug[:MAX_SLUG_LEN].rstrip("-")
    return slug or "untitled"


def content_hash(payload: Any) -> str:
    """Hash an API payload to detect real changes between syncs.

    Args:
        payload: The decoded API payload.

    Returns:
        A hex SHA-256 digest over the payload's canonical JSON form.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class SyncResult:
    """Per-note outcome of a sync pass."""

    note_id: str
    path: Path
    status: str  # "new" | "updated" | "unchanged"


class Archive:
    """Reads and writes the on-disk meeting archive."""

    def __init__(self, root: Path) -> None:
        """Initialize the archive.

        Args:
            root: Directory the archive lives in; created on first write.
                The path is resolved so that containment checks and
                ``relative_to`` comparisons operate on a single canonical form
                (on macOS, for example, ``/var`` resolves to ``/private/var``).
        """
        self.root = Path(root).expanduser().resolve()
        self._index: dict[str, dict[str, Any]] | None = None
        self._state: dict[str, Any] | None = None

    # -- index -------------------------------------------------------------

    @property
    def index_path(self) -> Path:
        """Path to the archive index.

        Returns:
            The ``index.json`` path.
        """
        return self.root / INDEX_NAME

    @property
    def state_path(self) -> Path:
        """Path to the sync state file.

        Returns:
            The ``.sync-state.json`` path.
        """
        return self.root / STATE_NAME

    @property
    def lock_path(self) -> Path:
        """Path to this archive's sync lock.

        The lock lives in the state directory rather than in the archive,
        for two reasons: a lock is runtime state and not archived content,
        and taking one must not bring an archive directory into existence --
        a sync that never starts, because the user is logged out, has to
        leave nothing behind.

        The digest is taken over the case-folded root. ``.resolve()`` fixes
        ``..`` and trailing slashes but leaves case alone, and on a
        case-insensitive filesystem -- the macOS default -- two spellings of
        one archive would otherwise take two locks and both syncs would run.
        ``os.path.normcase`` is a no-op outside Windows, so it cannot do
        this job. Folding unconditionally costs only a spurious "already
        running" for two *different* archives whose paths differ by case
        alone on a case-sensitive filesystem; a loud refusal is the better
        failure of the two.

        It remains a path identity, not an inode identity: the same
        directory reached through a macOS firmlink (a
        ``/System/Volumes/Data`` prefix) resolves to a different string and
        takes a different lock. ``st_dev``/``st_ino`` would settle it but
        needs the archive to exist, which would defeat not creating one for
        a sync that never starts.

        Returns:
            The lock path, named after a digest of the archive root so two
            different archives do not contend with each other.
        """
        from .mcp_auth import state_dir

        key = str(self.root).casefold()
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return state_dir() / f"{LOCK_NAME}-{digest}"

    @contextlib.contextmanager
    def exclusive(self) -> Iterator[None]:
        """Hold the archive for one writer, for the duration of the block.

        The index and sync state are read once per process and cached, so two
        syncs running at once each write a snapshot taken at their own start
        time: the second silently drops the entries and the watermark the
        first added. The files on disk survive, but the index that describes
        them and the watermark that decides what syncs next do not.

        ``flock`` is held by the open descriptor, so the kernel releases it
        however the process ends. That is why this is a lock file with no
        contents and no stale-lock recovery: there is no PID to go stale.

        Holding the lock is also the only moment at which temp files from a
        previous hard kill can safely be removed, so that happens here. See
        :meth:`sweep_stale_temp_files`.

        Raises:
            ArchiveBusy: Another process holds the archive.
        """
        lock_path = self.lock_path
        _secure_mkdir(lock_path.parent)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, FILE_MODE)
        try:
            os.chmod(fd, FILE_MODE)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                # Only contention means busy. A filesystem that cannot lock
                # (ENOLCK, EOPNOTSUPP on some network and FUSE mounts) must
                # not be reported as a second sync the user then goes looking
                # for.
                if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                    raise
                raise ArchiveBusy(self.root) from exc
            self.sweep_stale_temp_files()
            yield
        finally:
            os.close(fd)

    def sweep_stale_temp_files(self) -> int:
        """Delete atomic-write temp files a hard kill left behind.

        ``write_json`` removes its own temp file on any exception, but a
        ``SIGKILL``, a power loss or a lid close leaves one on disk. Each
        has a unique name, so they accumulate one per kill and no
        predictable path finds them; the old fixed ``<dest>.tmp`` name was
        self-limiting because the next write truncated it.

        The caller must hold :meth:`exclusive`. That is the whole reason
        this lives here rather than in ``write_json``: a sweep by one of two
        concurrent writers would delete the other's temp file out from under
        it, which is the crash the lock exists to prevent.

        Holding this archive's lock is not the same as holding every lock
        under this directory. A ``--profile`` archive nests inside the
        default one (``<archive_dir>/<profile>``), has its own lock, and may
        be syncing right now, so any nested archive root is pruned from the
        walk. Without that, a default sync deletes a profile sync's
        in-flight temp file and crashes it.

        Legacy fixed-name temps are matched too, so an archive written by an
        older build is cleaned on its next sync.

        Returns:
            How many files were removed.
        """
        if not self.root.is_dir():
            return 0
        removed = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            here = Path(dirpath)
            if here != self.root and (
                (here / STATE_NAME).exists() or (here / INDEX_NAME).exists()
            ):
                # Another archive's tree. Not ours to sweep, and it may be
                # mid-write behind its own lock.
                dirnames[:] = []
                continue
            for name in filenames:
                # Only this module's own temp shapes, so an unrelated file a
                # user parked in the archive is left alone.
                if not fnmatch(name, "*.json.tmp") and not fnmatch(
                    name, "*.json.*.tmp"
                ):
                    continue
                try:
                    (here / name).unlink()
                except OSError:
                    continue
                removed += 1
        return removed

    def load_index(self) -> dict[str, dict[str, Any]]:
        """Load the archive index, caching it for the process lifetime.

        Returns:
            A mapping of note id to its index entry; empty on first run or if
            the index is unreadable.
        """
        if self._index is None:
            self._index = _read_json(self.index_path, default={})
            if not isinstance(self._index, dict):
                self._index = {}
        return self._index

    def save_index(self) -> None:
        """Write the index to disk atomically."""
        _write_json(self.index_path, self.load_index())

    def load_state(self) -> dict[str, Any]:
        """Load sync state, caching it for the process lifetime.

        Returns:
            The sync state mapping; empty on first run.
        """
        if self._state is None:
            self._state = _read_json(self.state_path, default={})
            if not isinstance(self._state, dict):
                self._state = {}
        return self._state

    def save_state(self, **updates: Any) -> None:
        """Merge updates into the sync state and persist it.

        Args:
            **updates: Keys to set on the state mapping.
        """
        state = self.load_state()
        state.update(updates)
        _write_json(self.state_path, state)

    def source_state(self, source: str) -> dict[str, Any]:
        """Read the sync state belonging to one backend.

        Args:
            source: A source constant such as ``granola-mcp``.

        Returns:
            That source's state mapping; empty when it has never synced.
        """
        sources = self.load_state().get("sources")
        if not isinstance(sources, dict):
            return {}
        value = sources.get(source)
        return value if isinstance(value, dict) else {}

    def save_source_state(self, source: str, **updates: Any) -> None:
        """Merge updates into one backend's sync state and persist.

        The public API's ``updated_after`` is additionally mirrored to the
        legacy top-level key, so an archive written by this build is still
        readable by an older one.

        Args:
            source: The source these updates belong to.
            **updates: Keys to set on that source's state mapping.
        """
        state = self.load_state()
        sources = state.get("sources")
        if not isinstance(sources, dict):
            sources = {}
        entry = sources.get(source)
        if not isinstance(entry, dict):
            entry = {}
        entry.update(updates)
        sources[source] = entry
        state["sources"] = sources
        state["schema_version"] = STATE_SCHEMA_VERSION
        if source == SOURCE_PUBLIC_API and "updated_after" in updates:
            state["updated_after"] = updates["updated_after"]
        _write_json(self.state_path, state)
        self._state = state

    @property
    def watermark(self) -> str | None:
        """The ``updated_at`` high-water mark from the last public API sync.

        Reads the namespaced location first and falls back to the legacy
        top-level key, so an archive written before per-source state existed
        keeps its watermark instead of silently triggering a full backfill.

        Returns:
            An ISO 8601 timestamp, or ``None`` if no sync has completed.
        """
        value = self.source_state(SOURCE_PUBLIC_API).get("updated_after")
        if not value:
            value = self.load_state().get("updated_after")
        return str(value) if value else None

    # -- account identity --------------------------------------------------

    @property
    def account_email(self) -> str:
        """The Granola account this archive belongs to.

        Returns:
            The recorded email, or ``""`` for an archive that has never been
            claimed -- either because it is empty or because it predates this
            key.
        """
        account = self.load_state().get("account")
        if not isinstance(account, dict):
            return ""
        return str(account.get("email") or "")

    def claim_account(self, email: str, source: str) -> None:
        """Record which account this archive holds.

        Args:
            email: The account's email address.
            source: The backend the identity was learned from.
        """
        if not email:
            return
        account = self.load_state().get("account")
        first_seen = (
            account.get("first_seen")
            if isinstance(account, dict) and account.get("email") == email
            else None
        )
        self.save_state(
            account={
                "email": email,
                "source": source,
                "first_seen": first_seen or datetime.now().astimezone().isoformat(),
            }
        )

    def infer_account_email(self, sample: int = ACCOUNT_SAMPLE) -> str:
        """Work out whose meetings an unclaimed archive already holds.

        Archives written before the account key existed still carry the answer
        in their payloads, so an upgrade does not have to trust the next sync
        to be the right account. The most common email wins rather than the
        first found: a workspace-visible meeting created by a colleague would
        otherwise read as an account change.

        Args:
            sample: How many of the most recently archived notes to read.

        Returns:
            The dominant email, or ``""`` when the archive holds none -- an
            MCP archive whose elements carry no creator marker, for instance.
        """
        entries = sorted(
            self.load_index().values(),
            key=lambda e: str(e.get("archived_at") or ""),
            reverse=True,
        )[:sample]

        seen: Counter[str] = Counter()
        for entry in entries:
            try:
                path = _resolve_within(self.root, str(entry.get("path", "")), RAW_NAME)
            except UnsafeArchivePathError:
                continue
            raw = _read_json(path, default=None)
            if isinstance(raw, dict):
                email = _owner_email(raw)
                if email:
                    seen[email] += 1
        return seen.most_common(1)[0][0] if seen else ""

    # -- paths -------------------------------------------------------------

    def note_dir(self, note: Note) -> Path:
        """Compute the archive directory for a note.

        Args:
            note: The parsed note.

        Returns:
            A ``<root>/YYYY/MM/YYYY-MM-DD--slug--<id>`` path. Notes without a
            creation timestamp are filed under ``undated/``.

        Raises:
            UnsafeArchivePathError: If the note id is not a well-formed
                ``not_*`` or ``mcp_*`` key, or the computed path would escape
                the archive.
        """
        created = note.created_at
        slug = slugify(note.display_title)
        if note.id and not is_valid_archive_key(note.id):
            raise UnsafeArchivePathError(f"malformed note id: {note.id!r}")
        stem = f"{slug}--{note.id}" if note.id else slug
        if created is None:
            return _resolve_within(self.root, "undated", stem)
        return _resolve_within(
            self.root,
            f"{created.year:04d}",
            f"{created.month:02d}",
            f"{created:%Y-%m-%d}--{stem}",
        )

    # -- writing -----------------------------------------------------------

    def is_unchanged(self, note_id: str, digest: str) -> bool:
        """Check whether a note's payload is byte-identical to the archived one.

        Args:
            note_id: The public API note id.
            digest: The freshly computed content hash.

        Returns:
            ``True`` if the note is already archived with this exact payload
            and its directory still exists on disk.
        """
        entry = self.load_index().get(note_id)
        if not entry or entry.get("content_hash") != digest:
            return False
        path = entry.get("path")
        if not path:
            return False
        try:
            return _resolve_within(self.root, str(path)).is_dir()
        except UnsafeArchivePathError:
            # A tampered or corrupted index must not be trusted to point
            # outside the archive; treat it as "not archived" and re-fetch.
            return False

    def write_note(
        self,
        note: Note,
        note_md: str,
        transcript_md: str | None,
        *,
        source: str = SOURCE_PUBLIC_API,
        digest: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> SyncResult:
        """Write a note's raw payload and rendered Markdown into the archive.

        The verbatim payload is written first so that a rendering failure can
        never cost data. If the note's computed path has moved -- because its
        title or date changed -- the previous directory is relocated rather
        than left behind as a duplicate.

        Args:
            note: The parsed note.
            note_md: Rendered ``note.md`` contents.
            transcript_md: Rendered ``transcript.md`` contents, or ``None``
                when the note has no transcript.
            source: Provenance recorded on the index entry.
            digest: Content hash to record. Defaults to hashing the whole
                payload; the MCP backend passes a hash over the verbatim tool
                output alone, so derived fields cannot trigger a rewrite.
            extra: Source-specific keys merged into the index entry.

        Returns:
            The outcome for this note.
        """
        index = self.load_index()
        previous = index.get(note.id)
        target = self.note_dir(note)

        if previous and previous.get("path"):
            try:
                old = _resolve_within(self.root, str(previous["path"]))
            except UnsafeArchivePathError:
                # Never relocate based on an index entry pointing outside the
                # archive; fall through and write the note fresh at `target`.
                old = None
            if old is not None and old.is_dir() and old != target:
                _secure_mkdir(target.parent)
                shutil.move(str(old), str(target))

        _secure_mkdir(target)
        # Archives written before secure_mkdir tightened every level it
        # created have 0755 year and month directories. Tightening the ones
        # above this note lets an existing archive converge as it is written.
        for level in target.relative_to(self.root).parents:
            _secure_mkdir(self.root / level)

        _write_json(target / RAW_NAME, note.raw)
        _secure_write_text(target / NOTE_NAME, note_md)

        transcript_file = target / TRANSCRIPT_NAME
        if transcript_md:
            _secure_write_text(transcript_file, transcript_md)
        elif transcript_file.exists():
            # A transcript we already archived must survive upstream deletion.
            pass

        entry: dict[str, Any] = {
            "path": str(target.relative_to(self.root)),
            "title": note.display_title,
            "uuid": note.uuid,
            "created_at": note.created_at.isoformat() if note.created_at else None,
            "updated_at": note.updated_at.isoformat() if note.updated_at else None,
            "folders": [f.name for f in note.folder_membership],
            "has_transcript": bool(note.transcript),
            "content_hash": digest if digest is not None else content_hash(note.raw),
            "archived_at": datetime.now().astimezone().isoformat(),
            "upstream_missing": False,
            "source": source,
        }
        if note.degraded:
            entry["degraded"] = True
        if extra:
            entry.update(extra)
        index[note.id] = entry
        self._index = index
        return SyncResult(
            note_id=note.id,
            path=target,
            status="updated" if previous else "new",
        )

    def archived_source(self, note_id: str) -> str:
        """The provenance recorded for an archived note.

        Entries written before provenance tracking carry no ``source`` field
        and are, correctly, public API notes.

        Args:
            note_id: An archive key.

        Returns:
            The source constant for that note.
        """
        entry = self.load_index().get(note_id) or {}
        return str(entry.get("source") or SOURCE_PUBLIC_API)

    def archived_raw(self, note: Note) -> dict[str, Any] | None:
        """Read the ``raw.json`` this archive already holds for a note.

        Looks where the index says the note lives, then where ``write_note``
        would put it. A relocation interrupted before the index was saved
        leaves the files at the second place while the index still names the
        first, and an archived payload must not be mistaken for absent.

        Args:
            note: The note about to be written.

        Returns:
            The decoded payload, or ``None`` when there is no readable one.

        Raises:
            UnsafeArchivePathError: If the note id is malformed, exactly as
                ``write_note`` would for the same note.
        """
        entry = self.load_index().get(note.id) or {}
        directories = []
        if entry.get("path"):
            try:
                directories.append(_resolve_within(self.root, str(entry["path"])))
            except UnsafeArchivePathError:
                # Never read through an entry pointing outside the archive.
                pass
        directories.append(self.note_dir(note))
        for directory in directories:
            raw = _read_json(directory / RAW_NAME, default=None)
            if isinstance(raw, dict):
                return raw
        return None

    def uuid_index(self) -> dict[str, list[str]]:
        """Map each archived note's UUID to the archive key(s) holding it.

        A UUID mapping to more than one key means the same meeting is archived
        twice, which is the invariant ``verify`` reports on and the reason
        ``adopt_mcp_entry`` exists.

        Returns:
            A mapping of UUID to the archive keys carrying it.
        """
        mapping: dict[str, list[str]] = {}
        for note_id, entry in self.load_index().items():
            uuid = entry.get("uuid")
            if isinstance(uuid, str) and uuid:
                mapping.setdefault(uuid.lower(), []).append(note_id)
        return mapping

    def adopt_mcp_entry(self, note: Note) -> str | None:
        """Retire an MCP-sourced entry that the public API now covers.

        Seeds the ``not_*`` index entry with the MCP entry's ``path`` so that
        ``write_note``'s existing, containment-checked relocation moves the
        directory instead of leaving a duplicate behind, then drops the
        ``mcp_*`` key.

        Args:
            note: A freshly fetched public API note.

        Returns:
            The retired archive key, or ``None`` when nothing was adopted.
        """
        if not note.uuid or not note.id:
            return None
        index = self.load_index()
        if note.id in index:
            return None
        for key in list(index):
            entry = index[key]
            if key == note.id or not key.startswith("mcp_"):
                continue
            if str(entry.get("uuid") or "").lower() != note.uuid:
                continue
            path = entry.get("path")
            if path:
                index[note.id] = {"path": path}
            del index[key]
            self._index = index
            return key
        return None

    def mark_upstream_missing(
        self,
        note_ids: set[str],
        *,
        source: str | None = None,
        scope: set[str] | None = None,
    ) -> list[str]:
        """Flag archived notes that no longer appear upstream.

        Nothing is deleted. The flag records that Granola no longer serves the
        note, which is exactly the case the archive exists to protect against.

        Args:
            note_ids: Ids seen in the current full listing.
            source: When given, only entries from this source are considered.
                Without it, a full MCP scan would flag every public API note as
                missing, since the two backends see different id namespaces.
            scope: When given, only these keys are considered; anything else
                is neither flagged nor cleared. For a listing that is only
                trustworthy across part of the history, like the MCP's.

        Returns:
            The ids newly flagged as missing.
        """
        index = self.load_index()
        newly_missing = []
        for note_id, entry in index.items():
            if source is not None and self.archived_source(note_id) != source:
                continue
            if scope is not None and note_id not in scope:
                continue
            missing = note_id not in note_ids
            if missing and not entry.get("upstream_missing"):
                entry["upstream_missing"] = True
                entry["missing_since"] = datetime.now().astimezone().isoformat()
                newly_missing.append(note_id)
            elif not missing and entry.get("upstream_missing"):
                entry["upstream_missing"] = False
                entry.pop("missing_since", None)
        self._index = index
        return newly_missing
