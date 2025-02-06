#!/usr/bin/env python3
import argparse
import sqlite3
import sys
from common import get_round_prompt, get_confusion_matrix, get_printable_confusion_matrix_and_examples
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
            answer += print("# Historical results:")
            for r_id, r_prompt in history_rounds:
                hist_matrix = get_confusion_matrix(conn, r_id, example_count=0)
                answer += get_printable_confusion_matrix_and_examples(r_id, r_prompt, hist_matrix, show_examples=False)
        else:
            #print("No previous rounds found.")
            pass
    answer += "\n\n---------------------\n\n"
    answer += """Supply your answer in JSON format like this:

{
    "reasoning": "...",
    "updated_prompt": "..."
}

    Where `reasoning` explains why you are making the change and `updated_prompt` is the prompt that you think we should run next.
"""
    return answer

def run_reprompt(conn, prompting_creation_prompt, old_round_id, model):
    command = ["ollama", "run", model, "--format=json", "--verbose"]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True  # This ensures text mode instead of bytes
    )
    # Send the prompt and get outputs
    stdout, stderr = process.communicate(input=prompting_creation_prompt)
    answer = json.loads(stdout)
    if 'updated_prompt' not in answer:
        sys.exit(f"No updated prompt supplied: {answer}")
    info_start = stderr.index("total duration:")
    stderr = stderr[info_start:]
    split_id = get_split_id(conn, old_round_id)
    cur = conn.cursor()
    cur.execute("insert into rounds (split_id, prompt, reasoning_for_this_prompt, stderr_from_prompt_creation) values (?,?,?) returning round_id",
                [split_id, answer['updated_prompt'], answer['reasoning'], stderr])
    row = cur.fetchone()
    if row is None:
        sys.exit("Failed to create a new round")
    new_round_id = row[0]
    return (new_round_id, answer)


def main():
    parser = argparse.ArgumentParser(description="Show confusion matrix for a round")
    parser.add_argument('--database', default='titanic_medical.sqlite', help="Path to the SQLite database file")
    parser.add_argument('--round-id', type=int, required=True, help="Round ID")
    parser.add_argument('--example-count', type=int, default=3, help="Number of examples per cell")
    parser.add_argument('--show-history', type=int, default=2, help="Show confusion matrices for the previous N rounds")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model", default="llama3.3:latest", help="Model to do the retraining, not the model that does the inference")
    args = parser.parse_args()

    conn = sqlite3.connect(args.database)
    prompting_creation_prompt = get_prompt_for_updating_model(conn, args.round_id, args.example_count, args.show_history)
    if args.dry_run or args.verbose:
        print(prompting_creation_prompt)
    if args.dry_run:
        sys.exit(0)
    new_round_id, details = run_reprompt(conn, prompting_creation_prompt, args.round_id, args.model)
    if args.verbose:
        print("REASONING: {details['reasoning']}")
        print()
        print("NEW PROMPT: {details['updated_prompt']}")
        print()
        print("NEW ROUND ID: {new_round_id}")

if __name__ == '__main__':
    main()
