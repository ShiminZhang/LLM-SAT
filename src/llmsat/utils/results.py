"""Shared helpers for evaluation-result handling.

This module is the single source of truth for conventions that were
previously re-implemented (divergently) across collectors.
"""


def instance_key(filename: str) -> str:
    """Canonical instance key for a CNF filename or solving-log filename.

    The key is the basename truncated at its first dot. benchmark-database.de
    filenames begin with the instance's unique md5 hash, so truncation is
    collision-free, and every existing artifact (instance_categories.json,
    baseline_solving_times.json, solving_times_*.json) already uses this form.
    Handles "<name>.cnf", "<name>.normalised.cnf", and
    "<name>[.normalised].cnf.solving.log" alike.
    """
    base = filename.rsplit("/", 1)[-1]
    return base.split(".", 1)[0]
