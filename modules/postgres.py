#!/usr/bin/env python3
"""Utility helpers for PostgreSQL connections."""
import json
import os
from typing import Optional

import psycopg2


def _load_dsn_from_config(path: str) -> Optional[str]:
    """Load a DSN from a JSON config file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("postgres_dsn")
    except Exception:
        return None


def get_connection(dsn: Optional[str] = None, config_file: Optional[str] = None):
    """Return a psycopg2 connection using a DSN, config file or libpq defaults."""
    dsn = dsn or os.environ.get("POSTGRES_DSN")
    if not dsn:
        config_path = config_file or os.environ.get("POSTGRES_CONFIG")
        if config_path:
            dsn = _load_dsn_from_config(config_path)
    # Fall back to libpq environment variables and defaults if no DSN is found
    return psycopg2.connect(dsn or "")
