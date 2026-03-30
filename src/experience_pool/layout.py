"""Filesystem layout helpers for experience pool storage."""

from __future__ import annotations

from pathlib import Path

from .types import OutcomeLabel, PoolName


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"
"""Default root directory for all experience records and FAISS indices."""


def get_data_root(data_root: str | Path | None = None) -> Path:
    """Return a normalized data root path.

    Args:
        data_root: Optional custom root directory.

    Returns:
        Path: Existing directory path used for experience storage.
    """

    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_partition_dir(data_root: str | Path, pool_name: PoolName, outcome: OutcomeLabel) -> Path:
    """Return partition directory path and ensure it exists.

    Args:
        data_root: Root directory for experience storage.
        pool_name: Experience pool name.
        outcome: Outcome partition (`good` or `bad`).

    Returns:
        Path: Existing partition directory path.
    """

    partition_dir = get_data_root(data_root) / pool_name / outcome.value
    partition_dir.mkdir(parents=True, exist_ok=True)
    return partition_dir


def records_file_path(data_root: str | Path, pool_name: PoolName, outcome: OutcomeLabel) -> Path:
    """Return JSON file path for record payload storage.

    Args:
        data_root: Root directory for experience storage.
        pool_name: Experience pool name.
        outcome: Outcome partition.

    Returns:
        Path: Absolute file path to `records.json`.
    """

    return get_partition_dir(data_root, pool_name, outcome) / "records.json"


def index_file_path(data_root: str | Path, pool_name: PoolName, outcome: OutcomeLabel) -> Path:
    """Return FAISS index file path.

    Args:
        data_root: Root directory for experience storage.
        pool_name: Experience pool name.
        outcome: Outcome partition.

    Returns:
        Path: Absolute file path to `index.faiss`.
    """

    return get_partition_dir(data_root, pool_name, outcome) / "index.faiss"


def id_map_file_path(data_root: str | Path, pool_name: PoolName, outcome: OutcomeLabel) -> Path:
    """Return JSON file path for vector-position to record-id mapping.

    Args:
        data_root: Root directory for experience storage.
        pool_name: Experience pool name.
        outcome: Outcome partition.

    Returns:
        Path: Absolute file path to `id_map.json`.
    """

    return get_partition_dir(data_root, pool_name, outcome) / "id_map.json"
