#!/usr/bin/env python3
import argparse
import sqlite3
import sys
from common import get_round_prompt, get_patient_features
import subprocess
import json
import string
import llmcall

def predict(conn, round_id, patient_id, model='phi4:latest', dry_run=False):

    # Use common code to get the round instructions.
    instructions = get_round_prompt(conn, round_id)
    if instructions is None:
        sys.exit(f"Round ID {round_id} not found.")

    cur = conn.cursor()
    # Check to see if it exists already
    cur.execute("select count(*) from inferences where round_id = ? and patient_id = ?",
                [round_id, patient_id])
    row = cur.fetchone()
    if row[0] == 1:
        sys.exit(f"Patient {patient_id} has already been evaluated in round {round_id}")

    patient_features = get_patient_features(conn, patient_id)

    prompt = f"""This is an experiment in identifying whether an LLM can predict medical outcomes. Use the following methodology for predicting the outcome for this patient.

```
{instructions}
```

Patient Data:
{patient_features}
"""

    if dry_run:
        print(prompt)
        return

    try:
        prediction_output, run_info = llmcall.dispatch_prediction_prompt(model, prompt, ['Success', 'Failure'])
        cur.execute("insert into inferences (round_id, patient_id, narrative_text, llm_stderr, prediction) values (?, ?, ?, ?, ?)",
                    (round_id, patient_id, prediction_output['narrative_text'], run_info, prediction_output['prediction']))
        conn.commit()
    except KeyError:
        sys.stderr.write("There was a field missing in ChatGPT's response. Skipping.\n")
        # we'll pick it up next time through the loop, I guess
        pass



if __name__ == '__main__':
    default_database = os.environ.get('NARRATIVE_LEARNING_DATABASE', 'titanic_medical.sqlite')
    default_model = os.environ.get('NARRATIVE_LEARNING_INFERENCE_MODEL', 'phi4:latest')
    parser = argparse.ArgumentParser(
        description="Show the round prompt and patient data (excluding Decodex, Outcome, and PatientID)"
    )
    parser.add_argument('--database', default=default_database, help="Path to the SQLite database file")
    parser.add_argument('--round-id', type=int, required=True, help="Round ID")
    parser.add_argument('--patient-id', required=True, help="Patient ID")
    parser.add_argument("--dry-run", action="store_true", help="Just show the prompt, then exit")
    parser.add_argument("--model", default=default_model)
    # maybe I should check an environment variable for the default model
    args = parser.parse_args()

    conn = sqlite3.connect(args.database)

    predict(conn, args.round_id, args.patient_id, args.model, args.dry_run)
