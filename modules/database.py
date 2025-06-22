#!/usr/bin/env python3
import sqlite3
import sys
from typing import Any, List, Optional

def _execute(cur, conn, query: str, params: tuple = ()):  # type: ignore
    """Execute a query with placeholder adaptation for PostgreSQL."""
    if not isinstance(conn, sqlite3.Connection):
        query = query.replace("?", "%s")
    cur.execute(query, params)

from modules.exceptions import NonexistentRoundException

def get_database_path(conn: sqlite3.Connection) -> Optional[str]:
    """Get the file path of an SQLite database from its connection object."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA database_list")
    # The database info is returned as (seq, name, file)
    db_info = cursor.fetchall()

    # The main database is usually the one with name 'main'
    for entry in db_info:
        if entry[1] == 'main':
            return entry[2]

    # If we didn't find one labeled 'main', return the first file path
    if db_info:
        return db_info[0][2]

    return None

def get_round_prompt(conn: Any, round_id: int, database_path: Optional[str] = None, dataset: str = "") -> str:
    """
    Retrieve the prompt text for a given round.

    Args:
        conn: SQLite database connection
        round_id: ID of the round
        database_path: Path to the database file (for error reporting)

    Returns:
        Prompt text
    """
    cur = conn.cursor()
    table = f"{dataset}_rounds" if dataset else "rounds"
    _execute(cur, conn, f"SELECT prompt FROM {table} WHERE round_id = ?", (int(round_id),))
    row = cur.fetchone()

    if row is None:
        raise NonexistentRoundException(round_id, database_path)

    return row[0]
    
def get_round_reasoning(conn: Any, round_id: int, database_path: Optional[str] = None, dataset: str = "") -> str:
    """
    Retrieve the reasoning for the prompt text for a given round.

    Args:
        conn: SQLite database connection
        round_id: ID of the round
        database_path: Path to the database file (for error reporting)

    Returns:
        Reasoning text for the prompt
    """
    cur = conn.cursor()
    table = f"{dataset}_rounds" if dataset else "rounds"
    _execute(cur, conn, f"SELECT reasoning_for_this_prompt FROM {table} WHERE round_id = ?", (int(round_id),))
    row = cur.fetchone()

    if row is None:
        raise NonexistentRoundException(round_id, database_path)

    return row[0] if row[0] is not None else ""

def get_split_id(conn: Any, round_id: int, database_path: Optional[str] = None, dataset: str = "") -> int:
    """
    Get the split ID associated with a round.

    Args:
        conn: SQLite database connection
        round_id: ID of the round
        database_path: Path to the database file (for error reporting)

    Returns:
        Split ID
    """
    cur = conn.cursor()
    table = f"{dataset}_rounds" if dataset else "rounds"
    _execute(cur, conn, f"SELECT split_id FROM {table} WHERE round_id = ?", (int(round_id),))
    row = cur.fetchone()

    if row is None:
        raise NonexistentRoundException(round_id, database_path)

    return row[0]

def get_latest_split_id(conn: Any, dataset: str = "", investigation_id: int | None = None) -> int:
    """
    Get the split_id from the most recent round.

    Args:
        conn: SQLite database connection

    Returns:
        Integer split ID

    Raises:
        SystemExit: If no rounds are found in the database
    """
    cur = conn.cursor()
    table = f"{dataset}_rounds" if dataset else "rounds"
    query = f"SELECT split_id FROM {table}"
    params: list[Any] = []
    if investigation_id is not None:
        query += " WHERE investigation_id = ?"
        params.append(investigation_id)
    query += " ORDER BY round_id DESC LIMIT 1"
    _execute(cur, conn, query, tuple(params))
    row = cur.fetchone()
    if row is None:
        sys.exit("No rounds found in database")
    split_id = row[0]
    return split_id

def get_rounds_for_split(conn: Any, split_id: int, dataset: str = "", investigation_id: int | None = None) -> List[int]:
    """
    Get all round IDs for a given split_id.

    Args:
        conn: SQLite database connection
        split_id: The split ID to query

    Returns:
        List of round IDs
    """
    cur = conn.cursor()
    table = f"{dataset}_rounds" if dataset else "rounds"
    query = f"SELECT round_id FROM {table} WHERE split_id = ?"
    params: list[Any] = [split_id]
    if investigation_id is not None:
        query += " AND investigation_id = ?"
        params.append(investigation_id)
    query += " ORDER BY round_id"
    _execute(cur, conn, query, tuple(params))
    rounds = [row[0] for row in cur.fetchall()]
    return rounds

def get_processed_rounds_for_split(conn: Any, split_id: int, dataset: str = "", investigation_id: int | None = None) -> List[int]:
    """
    Get all round IDs for a given split_id that have inferences.

    Args:
        conn: SQLite database connection
        split_id: The split ID to query

    Returns:
        List of round IDs that have inferences
    """
    cur = conn.cursor()
    inf_table = f"{dataset}_inferences" if dataset else "inferences"
    answer = []
    for r in get_rounds_for_split(conn, split_id, dataset, investigation_id):
        query = f"select count(*) from {inf_table} where round_id = ?"
        params: list[Any] = [r]
        if investigation_id is not None:
            query += " and investigation_id = ?"
            params.append(investigation_id)
        _execute(cur, conn, query, tuple(params))
        row = cur.fetchone()
        if row[0] == 0:
            continue
        answer.append(r)
    return answer
