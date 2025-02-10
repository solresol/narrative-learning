#!/usr/bin/env python3
import argparse
import sqlite3
import sys
from common import get_round_prompt, get_patient_features
import subprocess
import json
import string

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

Output in JSON format like this:
""" + """
    {
        "narrative_text": "...",
        "prediction": "..."
    }

`narrative_text` is where you describe your thinking process in evaluating the prompt.
`prediction` is either Success or Failure
    """
    if dry_run:
        print(prompt)
        return

    command = ["ollama", "run", model, "--format=json", "--verbose"]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True  # This ensures text mode instead of bytes
    )

    # Send the prompt and get outputs
    stdout, stderr = process.communicate(input=prompt)
    print(stdout)
    answer = json.loads(stdout)
    if answer['prediction'] not in ['Success', 'Failure']:
        # We didn't do anything. Leave it for now, and hopefully we'll come
        # back in another round
        sys.stderr.write("Invalid prediction\n")
        return
    if 'narrative_text' not in answer:
        sys.stderr.write("No narrative text\n")
        answer['narrative_text'] = ''
    #print(f"stdout = {json.dumps(answer,indent=4)}")
    info_start = stderr.index("total duration:")
    stderr = stderr[info_start:]
    print(f"stderr = {stderr}")
    cur.execute("insert into inferences (round_id, patient_id, narrative_text, llm_stderr, prediction) values (?, ?, ?, ?, ?)",
                   (round_id, patient_id, answer['narrative_text'], stderr, answer['prediction']))
    conn.commit()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Show the round prompt and patient data (excluding Decodex, Outcome, and PatientID)"
    )
    parser.add_argument('--database', default='titanic_medical.sqlite', help="Path to the SQLite database file")
    parser.add_argument('--round-id', type=int, required=True, help="Round ID")
    parser.add_argument('--patient-id', required=True, help="Patient ID")
    parser.add_argument("--dry-run", action="store_true", help="Just show the prompt, then exit")
    parser.add_argument("--model", default="phi4:latest")
    # maybe I should check an environment variable for the default model
    args = parser.parse_args()

    conn = sqlite3.connect(args.database)

    predict(conn, args.round_id, args.patient_id, args.model, args.dry_run)
