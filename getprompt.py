#!/usr/bin/env python3

import sqlite3
import os
import json

if __name__ == '__main__':
    default_database = os.environ.get('NARRATIVE_LEARNING_DATABASE', None)
    default_config = os.environ.get('NARRATIVE_LEARNING_CONFIG', None)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', default=default_database,
                        required = default_database is None, help="Path to the SQLite database file")
    parser.add_argument('--round-id', type=int,  help="Round ID")
    args = parser.parse_args()

    conn = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    cur = conn.cursor()
    if args.round_id:
        cur.execute("SELECT prompt FROM rounds WHERE round_id = ?", (args.round_id,))
    else:
        cur.execute("select prompt from rounds order by round_id desc limit 1")
    row = cur.fetchone()

    if row is None:
        sys.exit(f"Round ID {round_id} not found")

    encoded_prompt = row[0]
    print(encoded_prompt)
