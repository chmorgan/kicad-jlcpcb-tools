"""Consolidate legacy global plugin databases into a single global.db."""

from __future__ import annotations

from collections.abc import Sequence
import contextlib
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sqlite3
from typing import Optional

GLOBAL_DB_NAME = "global.db"
CORRECTIONS_LEGACY_NAME = "corrections.db"
MAPPINGS_LEGACY_NAME = "mappings.db"

# Migrate in a fixed order so logs and partial recovery stay deterministic.
_LEGACY_SOURCES: Sequence[tuple[str, str]] = (
    (CORRECTIONS_LEGACY_NAME, "corrections"),
    (MAPPINGS_LEGACY_NAME, "part preferences"),
)


@dataclass(frozen=True)
class LegacyDatabaseOutcome:
    """Result of attempting to migrate one legacy global database file."""

    filename: str
    label: str
    status: str  # "absent", "migrated", or "failed"
    error: Optional[str] = None  # noqa: UP045


@dataclass(frozen=True)
class MigrationResult:
    """Outcomes for every legacy global database considered during startup."""

    outcomes: tuple[LegacyDatabaseOutcome, ...]

    def uses_global(self, filename: str) -> bool:
        """Return True when that legacy file is gone or was never present."""
        for outcome in self.outcomes:
            if outcome.filename == filename:
                return outcome.status in {"absent", "migrated"}
        return True


def global_db_path(datadir: str) -> str:
    """Return the path of the consolidated global plugin database."""
    return os.path.join(datadir, GLOBAL_DB_NAME)


def legacy_db_path(datadir: str, filename: str) -> str:
    """Return the path of a legacy global database under datadir."""
    return os.path.join(datadir, filename)


def migrate_legacy_global_databases(
    datadir: str,
    logger: Optional[logging.Logger] = None,  # noqa: UP045
) -> MigrationResult:
    """Copy each present legacy global DB into global.db, then remove the source.

    Migration is independent per legacy file. A failure leaves that file in place
    for the next startup while other files may still migrate successfully. When a
    legacy file remains after a previous successful copy, tables from that source
    are replaced from the legacy file again before it is removed.
    """
    log = logger if logger is not None else logging.getLogger(__name__)
    destination = Path(global_db_path(datadir))
    log.info(
        "Checking for legacy global databases to migrate into %s",
        destination,
    )
    outcomes: list[LegacyDatabaseOutcome] = []
    for filename, label in _LEGACY_SOURCES:
        outcomes.append(_migrate_one_legacy_database(datadir, filename, label, log))
    migrated = [item.filename for item in outcomes if item.status == "migrated"]
    failed = [item.filename for item in outcomes if item.status == "failed"]
    if not migrated and not failed:
        log.info(
            "No legacy global databases found; using %s for corrections and "
            "part preferences.",
            destination,
        )
    else:
        log.info(
            "Global database migration finished. Migrated: %s. Failed: %s. "
            "Destination: %s",
            ", ".join(migrated) if migrated else "none",
            ", ".join(failed) if failed else "none",
            destination,
        )
    return MigrationResult(outcomes=tuple(outcomes))


def _migrate_one_legacy_database(
    datadir: str,
    filename: str,
    label: str,
    log: logging.Logger,
) -> LegacyDatabaseOutcome:
    """Migrate one legacy file into global.db or report why it remains."""
    source = Path(legacy_db_path(datadir, filename))
    destination = Path(global_db_path(datadir))
    if not source.exists():
        log.info("Legacy %s database %s not found — nothing to migrate.", label, source)
        return LegacyDatabaseOutcome(filename, label, "absent")

    log.info(
        "Found legacy %s database %s — migrating tables into %s",
        label,
        source,
        destination,
    )
    try:
        _copy_legacy_database(source, destination, log)
        _remove_sqlite_files(source, log)
    except (OSError, sqlite3.Error) as error:
        log.error(
            "Failed to migrate legacy %s database %s into %s: %s. "
            "Leaving the legacy file in place for the next startup.",
            label,
            source,
            destination,
            error,
        )
        return LegacyDatabaseOutcome(filename, label, "failed", str(error))

    log.info(
        "Migration of legacy %s database %s completed; removed the legacy file.",
        label,
        source,
    )
    return LegacyDatabaseOutcome(filename, label, "migrated")


def _copy_legacy_database(source: Path, destination: Path, log: logging.Logger) -> None:
    """Replace destination tables that exist in source, then commit atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(destination)) as dest:
        dest.execute("BEGIN IMMEDIATE")
        try:
            dest.execute(f"ATTACH DATABASE {_sql_literal(str(source))} AS legacy")
            tables = dest.execute(
                "SELECT name, sql FROM legacy.sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()
            if not tables:
                log.info(
                    "Legacy database %s has no user tables; removing it after "
                    "opening %s.",
                    source,
                    destination,
                )
            for name, create_sql in tables:
                if not create_sql:
                    raise sqlite3.DatabaseError(
                        f"legacy table {name!r} has no CREATE statement"
                    )
                existing = dest.execute(
                    "SELECT 1 FROM main.sqlite_master "
                    "WHERE type = 'table' AND name = ?",
                    (name,),
                ).fetchone()
                if existing:
                    log.warning(
                        "Replacing existing table %r in %s from legacy database %s",
                        name,
                        destination,
                        source,
                    )
                quoted = _quote_ident(name)
                dest.execute(f"DROP TABLE IF EXISTS main.{quoted}")
                dest.execute(create_sql)
                dest.execute(f"INSERT INTO main.{quoted} SELECT * FROM legacy.{quoted}")
                count = dest.execute(f"SELECT COUNT(*) FROM main.{quoted}").fetchone()[
                    0
                ]
                log.info("  copied table %r (%s rows)", name, count)
                for (index_sql,) in dest.execute(
                    "SELECT sql FROM legacy.sqlite_master "
                    "WHERE type = 'index' AND tbl_name = ? AND sql IS NOT NULL "
                    "ORDER BY name",
                    (name,),
                ):
                    dest.execute(index_sql)
                    log.info("  copied index for table %r", name)
            for (trigger_sql,) in dest.execute(
                "SELECT sql FROM legacy.sqlite_master "
                "WHERE type = 'trigger' AND sql IS NOT NULL "
                "ORDER BY name"
            ):
                dest.execute(trigger_sql)
                log.info("  copied trigger from legacy database")
            dest.commit()
        except BaseException:
            dest.rollback()
            raise
        finally:
            with contextlib.suppress(sqlite3.Error):
                dest.execute("DETACH DATABASE legacy")


def _remove_sqlite_files(path: Path, log: logging.Logger) -> None:
    """Delete a SQLite database and any journal/WAL sidecars."""
    sidecars = (
        path,
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
        Path(f"{path}-journal"),
    )
    for candidate in sidecars:
        if candidate.exists() or candidate.is_symlink():
            candidate.unlink()
            log.info("  removed %s", candidate)


def _quote_ident(name: str) -> str:
    """Quote a SQLite identifier."""
    return '"' + name.replace('"', '""') + '"'


def _sql_literal(value: str) -> str:
    """Quote a SQL string literal."""
    return "'" + value.replace("'", "''") + "'"
