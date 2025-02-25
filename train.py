#!/usr/bin/env python3
import argparse
import sqlite3
import sys
import json
import llmcall
import os

from common import get_round_prompt, get_confusion_matrix, get_printable_confusion_matrix_and_examples, get_split_id
import subprocess

def get_prompt_for_updating_model(conn, round_id, example_count, history_rounds):
    answer = """

You are part of a program that is trying to learn the causes for
success or failure of a medical intervention. At each round, a prompt
is shown to an LLM together with the each patient's medical details
one at a time. It then attempts to predict the outcome based on the
rules in the prompt. This process works well if the prompt has very
explicit and clear rules: aim for unambiguous thresholds for values,
clear criteria for labels and careful wording for and 'AND' or 'OR'
logic you need to express.

We would like to improve the prompt that is being used.

Please create a new prompt that will reduce the number of false
positives and false negatives in this dataset. You can see the
prompt(s) that have been used previously, and how effective they
were. There are also some examples of where those prompt(s) did and
didn't work.

----------------------------

"""
    prompt = get_round_prompt(conn, round_id)
    matrix = get_confusion_matrix(conn, round_id, example_count=example_count)
    answer += get_printable_confusion_matrix_and_examples(round_id, prompt, matrix, show_examples=True)

    # History rounds (without examples)
    if history_rounds > 0:
        cur = conn.cursor()
        cur.execute("""
            SELECT round_id, prompt
              FROM rounds
             WHERE round_id < ?
          ORDER BY round_id DESC
             LIMIT ?
        """, (round_id, history_rounds))
        history_rounds = cur.fetchall()

        if history_rounds:
            answer += "# Historical results:"
            for r_id, r_prompt in history_rounds:
                hist_matrix = get_confusion_matrix(conn, r_id, example_count=0)
                answer += get_printable_confusion_matrix_and_examples(r_id, r_prompt, hist_matrix, show_examples=False)
        else:
            #print("No previous rounds found.")
            pass
    answer += "\n\n---------------------\n\n"
    return answer

def run_reprompt(conn, prompting_creation_prompt, old_round_id, model):
    new_prompt, process_info = llmcall.dispatch_reprompt_prompt(model, prompting_creation_prompt)
    split_id = get_split_id(conn, old_round_id)
    cur = conn.cursor()
    cur.execute("insert into rounds (split_id, prompt, reasoning_for_this_prompt, stderr_from_prompt_creation) values (?,?,?,?) returning round_id",
                [split_id, new_prompt['updated_prompt'], new_prompt['reasoning'], process_info])
    row = cur.fetchone()
    if row is None:
        sys.exit("Failed to create a new round")
    new_round_id = row[0]
    conn.commit()
    return (new_round_id, new_prompt)


def main():
    default_database = os.environ.get('NARRATIVE_LEARNING_DATABASE', None)
    default_model = os.environ.get('NARRATIVE_LEARNING_TRAINING_MODEL', None)
    default_example_count = int(os.environ.get('NARRATIVE_LEARNING_EXAMPLE_COUNT', '3'))
    parser = argparse.ArgumentParser(description="Show confusion matrix for a round")
    parser.add_argument('--database', default=default_database, help="Path to the SQLite database file")
    parser.add_argument('--round-id', type=int, required=True, help="Round ID")
    parser.add_argument('--example-count', type=int, default=default_example_count, help="Number of examples per cell")
    parser.add_argument('--show-history', type=int, default=2, help="Show confusion matrices for the previous N rounds")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model", default=default_model, help="Model to do the retraining, not the model that does the inference")
    parser.add_argument("--round-tracking-file", help="A file that will have the new round ID written to it")
    args = parser.parse_args()

    if args.database is None:
        sys.exit("Must specify a database via --database or NARRATIVE_LEARNING_DATABASE env")
    if args.model is None:
        sys.exit("Must specify a model via --model or NARRATIVE_LEARNING_TRAINING_MODEL env")

    conn = sqlite3.connect(args.database)
    prompting_creation_prompt = get_prompt_for_updating_model(conn, args.round_id, args.example_count, args.show_history)
    if args.dry_run or args.verbose:
        print(prompting_creation_prompt)
    if args.dry_run:
        sys.exit(0)
    new_round_id, details = run_reprompt(conn, prompting_creation_prompt, args.round_id, args.model)
    if args.verbose:
        print(f"REASONING: {details['reasoning']}")
        print()
        print(f"NEW PROMPT: {details['updated_prompt']}")
        print()
        print(f"NEW ROUND ID: {new_round_id}")
    if args.round_tracking_file:
        with open(args.round_tracking_file, 'w') as f:
            f.write(f"{new_round_id}")

if __name__ == '__main__':
    main()
