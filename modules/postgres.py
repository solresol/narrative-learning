#!/usr/bin/env python3
"""Utility helpers for PostgreSQL connections."""
import json
import os
from typing import Optional, Tuple

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

    # If no DSN is supplied, rely on libpq environment defaults.
    return psycopg2.connect(dsn) if dsn else psycopg2.connect()


def get_investigation_settings(conn, investigation_id: int) -> Tuple[str, str]:
    """Return ``(dataset, config_file)`` for an investigation ID."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT i.dataset, d.config_file
          FROM investigations i
          JOIN datasets d ON i.dataset = d.dataset
         WHERE i.id = %s
        """,
        (investigation_id,),
    )
    row = cur.fetchone()
    if row is None:
        raise SystemExit(f"investigation {investigation_id} not found")
    return row[0], row[1]
