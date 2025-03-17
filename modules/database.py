#!/usr/bin/env python3
import sqlite3
import sys
from typing import Dict, List, Any, Optional, Tuple, Union

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

def get_round_prompt(conn: sqlite3.Connection, round_id: int, database_path: Optional[str] = None) -> str:
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
    cur.execute("SELECT prompt FROM rounds WHERE round_id = ?", (int(round_id),))
    row = cur.fetchone()

    if row is None:
        raise NonexistentRoundException(round_id, database_path)

    return row[0]
    
def get_round_reasoning(conn: sqlite3.Connection, round_id: int, database_path: Optional[str] = None) -> str:
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
    cur.execute("SELECT reasoning_for_this_prompt FROM rounds WHERE round_id = ?", (int(round_id),))
    row = cur.fetchone()

    if row is None:
        raise NonexistentRoundException(round_id, database_path)

    return row[0] if row[0] is not None else ""

def get_split_id(conn: sqlite3.Connection, round_id: int, database_path: Optional[str] = None) -> int:
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
    cur.execute("SELECT split_id FROM rounds WHERE round_id = ?", (round_id,))
    row = cur.fetchone()

    if row is None:
        raise NonexistentRoundException(round_id, database_path)

    return row[0]

def get_latest_split_id(conn: sqlite3.Connection) -> int:
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
    cur.execute("""
        SELECT split_id
        FROM rounds
        ORDER BY round_id DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    if row is None:
        sys.exit("No rounds found in database")
    split_id = row[0]
    return split_id

def get_rounds_for_split(conn: sqlite3.Connection, split_id: int) -> List[int]:
    """
    Get all round IDs for a given split_id.

    Args:
        conn: SQLite database connection
        split_id: The split ID to query

    Returns:
        List of round IDs
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT round_id
        FROM rounds
        WHERE split_id = ?
        ORDER BY round_id
    """, (split_id,))
    rounds = [row[0] for row in cur.fetchall()]
    return rounds

def get_processed_rounds_for_split(conn: sqlite3.Connection, split_id: int) -> List[int]:
    """
    Get all round IDs for a given split_id that have inferences.

    Args:
        conn: SQLite database connection
        split_id: The split ID to query

    Returns:
        List of round IDs that have inferences
    """
    cur = conn.cursor()
    answer = []
    for r in get_rounds_for_split(conn, split_id):
        cur.execute("select count(*) from inferences where round_id = ?", [r])
        row = cur.fetchone()
        if row[0] == 0:
            continue
        answer.append(r)
    return answer