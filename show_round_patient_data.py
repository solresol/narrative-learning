#!/usr/bin/env python3
import argparse
import sqlite3
import sys
from common import get_round_prompt

def main():
    parser = argparse.ArgumentParser(
        description="Show the round prompt and patient data (excluding Decodex, Outcome, and PatientID)"
    )
    parser.add_argument('--database', default='titanic_medical.sqlite', help="Path to the SQLite database file")
    parser.add_argument('--round-id', type=int, required=True, help="Round ID")
    parser.add_argument('--patient-id', required=True, help="Patient ID")
    args = parser.parse_args()

    conn = sqlite3.connect(args.database)

    # Use common code to get the round prompt.
    prompt = get_round_prompt(conn, args.round_id)
    if prompt is None:
        print(f"Round ID {args.round_id} not found.")
        sys.exit(1)

    # Retrieve patient data, excluding PatientID, Decodex, and Outcome.
    cur = conn.cursor()
    patient_query = """
    SELECT Treatment_Group, Sex, Treatment_Months, Genetic_Class_A_Matches,
           Genetic_Class_B_Matches, TcQ_mass, Cohort
      FROM medical_treatment_data
     WHERE Patient_ID = ?
    """
    cur.execute(patient_query, (args.patient_id,))
    patient_row = cur.fetchone()
    if not patient_row:
        print(f"Patient ID '{args.patient_id}' not found.")
        sys.exit(1)


    print("Round Prompt:")
    print(prompt)
    print("\nPatient Data:")

    columns = [
        "Group", "Sex", "Treatment_Months",
        "Genetic_Class_A_Matches", "Genetic_Class_B_Matches",
        "TcQ_mass", "Cohort"
    ]
    for col, value in zip(columns, patient_row):
        print(f"{col}: {value}")

if __name__ == '__main__':
    main()
