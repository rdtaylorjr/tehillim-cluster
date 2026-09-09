"""Writes an output file by rename, so an interrupted run keeps the previous file intact."""

import json
import os
from collections.abc import Callable
from pathlib import Path


def replace_atomically(path: Path, write: Callable[[Path], None]) -> None:
    """Writes through a sibling temp file and renames, so a failed write publishes nothing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        write(temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def write_text(path: Path, text: str) -> None:
    """Writes text, replacing the target only once the write succeeds."""

    def write(target: Path) -> None:
        """Discards the character count write_text returns, which the writer contract forbids."""
        target.write_text(text, encoding="utf-8")

    replace_atomically(path, write)


def write_json(path: Path, payload: object, *, compact: bool = False) -> None:
    """Writes JSON, replacing the target only once serialisation succeeds, and never with NaN."""
    #: allow_nan=False, because NaN is not JSON and no reader downstream accepts it.
    separators = (",", ":") if compact else None
    write_text(
        path, json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=separators)
    )
