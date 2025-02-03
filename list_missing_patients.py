#!/usr/bin/env python3
import argparse
import sqlite3
import sys

def main():
    parser = argparse.ArgumentParser(description="List patientIDs missing from inferences for a given round")
    parser.add_argument('--database', default='titanic_medical.sqlite', help="Path to the SQLite database file")
    parser.add_argument('--round-id', type=int, required=True, help="Round ID to check")
    args = parser.parse_args()

    try:
        conn = sqlite3.connect(args.database)
    except Exception as e:
        print(f"Failed to connect to database '{args.database}': {e}", file=sys.stderr)
        sys.exit(1)

    cur = conn.cursor()
    
    # Get patientIDs from medical_treatment_data that are NOT present in inferences for this round.
    query = """
    SELECT Patient_ID 
      FROM medical_treatment_data
     WHERE Patient_ID NOT IN (
           SELECT patient_id FROM inferences WHERE round_id = ?
     );
    """
    try:
        cur.execute(query, (args.round_id,))
        rows = cur.fetchall()
    except Exception as e:
        print(f"Database query failed: {e}", file=sys.stderr)
        sys.exit(1)

    if rows:
        for row in rows:
            print(row[0])
    else:
        print(f"No missing patientIDs found for round {args.round_id}.")
        sys.exit(2)

if __name__ == '__main__':
    main()
