"""
src/utils.py
============
Shared utility functions for logging, file I/O, and directory management.
"""

import json
import logging
from pathlib import Path
from typing import Any, Union


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a consistent, readable format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def save_json(data: dict[str, Any], path: Path) -> None:
    """Serialize a dictionary to JSON, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def load_json(path: Path) -> dict[str, Any]:
    """Load and parse a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dirs(dirs: list[Union[str, Path]]) -> None:
    """Create each directory (and any missing parents) if it does not exist."""
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
