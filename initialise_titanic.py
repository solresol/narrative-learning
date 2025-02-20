#!/usr/bin/env python3
import argparse
import sqlite3
import pandas as pd
import uuid
import sys
import random
import os

def main():
    default_database = os.environ.get('NARRATIVE_LEARNING_DATABASE', 'titanic_medical.sqlite')
    parser = argparse.ArgumentParser(
        description="Import Titanic CSV data into the medical treatment database."
    )
    parser.add_argument(
        "--database", default=default_database,
        help="Path to the SQLite database file."
    )
    parser.add_argument(
        "--schema", default="schema.sql",
        help="SQL file containing schema initialisation commands (default: schema.sql)."
    )
    parser.add_argument(
        "--source", default="titanic.csv",
        help="CSV source file (default: titanic.csv)."
    )
    parser.add_argument(
        "--prompt", default="Choose randomly",
        help="Prompt to insert into the rounds table (default: 'Choose randomly')."
    )
    parser.add_argument(
        "--holdout", default=0.2, type=float,
        help="Proportion of the data to mark as held out (never to be used in training)")

    parser.add_argument(
        "--validation", default=0.5, type=float,
        help="Proportion of the holdout data to use for validation")
    args = parser.parse_args()

    # Connect to the database.
    try:
        conn = sqlite3.connect(args.database)
    except Exception as e:
        print(f"Oops, failed to connect to database '{args.database}': {e}", file=sys.stderr)
        sys.exit(1)

    # Initialise the schema (this won't drop any existing tables because the schema file should use IF NOT EXISTS)
    try:
        with open(args.schema, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
    except Exception as e:
        print(f"Oops, failed to initialise schema from '{args.schema}': {e}", file=sys.stderr)
        sys.exit(1)

    # Load the source CSV.
    try:
        titanic_df = pd.read_csv(args.source)
    except Exception as e:
        print(f"Oops, failed to read CSV file '{args.source}': {e}", file=sys.stderr)
        sys.exit(1)

    # Build the medical_df by transforming titanic_df.
    # We'll construct a new DataFrame with the required columns.
    medical_df = pd.DataFrame()

    # 1. Use PassengerId as Decodex.
    medical_df['Decodex'] = titanic_df['PassengerId']

    # 2. Generate a unique PatientID.
    medical_df['Patient_ID'] = [str(uuid.uuid4()) for _ in range(len(titanic_df))]

    # 3. Outcome: Convert Survived (0,1) -> (Failure, Success)
    medical_df['Outcome'] = titanic_df['Survived'].map({0: 'Failure', 1: 'Success'})

    # 4. Group: Convert Pclass (1,2,3) -> (Beta, Omicron, Delta)
    medical_df['Treatment_Group'] = titanic_df['Pclass'].map({1: 'Beta', 2: 'Omicron', 3: 'Delta'})

    # 5. Drop Name from titanic_df (we don't need it further)
    titanic_df = titanic_df.drop(columns=['Name'])

    # 6. Sex: Invert Sex (male becomes female and vice versa)
    medical_df['Sex'] = titanic_df['Sex'].map({'male': 'female', 'female': 'male'})

    # 7. Treatment Months: Convert Age to Treatment Months (3 * Age), imputing missing with mean.
    mean_age = titanic_df['Age'].mean()
    medical_df['Treatment_Months'] = titanic_df['Age'].fillna(mean_age) * 3
    titanic_df = titanic_df.drop(columns=['Age'])

    # 8. Genetic Class A Matches: SibSp + 1.
    medical_df['Genetic_Class_A_Matches'] = titanic_df['SibSp'] + 1
    titanic_df = titanic_df.drop(columns=['SibSp'])

    # 9. Genetic Class B Matches: Parch + 1.
    medical_df['Genetic_Class_B_Matches'] = titanic_df['Parch'] + 1
    titanic_df = titanic_df.drop(columns=['Parch'])

    # 10. TcQ mass: Fare * 1000.
    medical_df['TcQ_mass'] = titanic_df['Fare'] * 1000
    titanic_df = titanic_df.drop(columns=['Fare'])

    # 11. Drop Cabin.
    titanic_df = titanic_df.drop(columns=['Cabin'])

    # 12. Cohort: Change Embarked (S, C, Q) -> (Melbourne, Delhi, Lisbon).
    medical_df['Cohort'] = titanic_df['Embarked'].map({'S': 'Melbourne', 'C': 'Delhi', 'Q': 'Lisbon'})
    titanic_df = titanic_df.drop(columns=['Embarked'])

    # Set Patient_ID as the index.
    medical_df.set_index('Patient_ID', inplace=True)

    cur = conn.cursor()
    cur.execute("insert into splits default values returning split_id")
    row = cur.fetchone()
    split_id = row[0]

    cur.execute("INSERT INTO rounds (prompt, split_id) VALUES (?,?)", (args.prompt,split_id))

    insert_sql = '''
    INSERT INTO medical_treatment_data (
      Patient_ID,
      Decodex,
      Outcome,
      Treatment_Group,
      Sex,
      Treatment_Months,
      "Genetic_Class_A_Matches",
      "Genetic_Class_B_Matches",
      TcQ_mass,
      Cohort
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
    for patient_id, row in medical_df.iterrows():
        cur.execute(insert_sql, (
            patient_id,
            row['Decodex'],
            row['Outcome'],
            row['Treatment_Group'],
            row['Sex'],
            row['Treatment_Months'],
            row['Genetic_Class_A_Matches'],
            row['Genetic_Class_B_Matches'],
            row['TcQ_mass'],
            row['Cohort']
        ))
        holdout = random.random() < args.holdout
        validation = holdout and (random.random() < args.validation)
        cur.execute("insert into patient_split (split_id, patient_id, holdout, validation) values (?,?,?, ?)",
                    (split_id, patient_id, holdout, validation))
    conn.commit()

    print("Database initialised successfully!")
    conn.close()

if __name__ == '__main__':
    main()
