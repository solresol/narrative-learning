#!/usr/bin/env python3
import argparse
import sqlite3
import sys
import predict

def main():
    parser = argparse.ArgumentParser(description="List patientIDs missing from inferences for a given round")
    parser.add_argument('--database', default='titanic_medical.sqlite', help="Path to the SQLite database file")
    parser.add_argument('--round-id', type=int, required=True, help="Round ID to check")
    parser.add_argument("--list", action="store_true", help="Just list the patients yet to be processed")
    parser.add_argument("--loop", action="store_true", help="Loop until there are no more patients to process")
    parser.add_argument("--stop-after", type=int, default=None, help="Stop after this many predictions")
    args = parser.parse_args()

    if args.list and args.loop:
        sys.exit("Can't loop if we aren't going to process anything")

    conn = sqlite3.connect(args.database)
    cur = conn.cursor()

    predictions_done = 0
    
    while True:
        # Get patientIDs from medical_treatment_data that are NOT present in inferences for this round.
        query = """
        SELECT Patient_ID 
        FROM medical_treatment_data
        WHERE Patient_ID NOT IN (
           SELECT patient_id FROM inferences WHERE round_id = ?
        );
        """
        cur.execute(query, (args.round_id,))
        anything_left = False
        for row in cur:
            anything_left = True
            if args.list:
                print(row[0])
                continue
            predict.predict(conn, args.round_id, row[0])
            predictions_done += 1
            if args.stop_after is not None and predictions_done >= args.stop_after:
                sys.exit(0)
        if not anything_left:
            if args.loop:
                # we succeeded
                sys.exit(0)
            # Otherwise, we weren't looping and there was nothing to do
            sys.exit(f"No missing patientIDs found for round {args.round_id}.")
        if args.loop:
            # Try again, in case there were errors
            continue
        break

if __name__ == '__main__':
    main()
