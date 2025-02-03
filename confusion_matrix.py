#!/usr/bin/env python3
import argparse
import sqlite3
import sys
from common import get_round_prompt, get_confusion_matrix

def print_confusion_matrix(round_id, prompt, matrix, show_examples=True):
    print(f"Round ID: {round_id}")
    print("Prompt:")
    print(prompt)
    print("\nConfusion Matrix:")
    # Layout: rows are Actual values; columns are Predicted.
    print(f"{'':15s} {'Predicted Positive':20s} {'Predicted Negative':20s}")
    # For actual positive:
    tp = matrix['TP']['count']
    fn = matrix['FN']['count']
    print(f"{'Actual Positive':15s} {tp:20d} {fn:20d}")
    # For actual negative:
    fp = matrix['FP']['count']
    tn = matrix['TN']['count']
    print(f"{'Actual Negative':15s} {fp:20d} {tn:20d}")
    print()

    if show_examples:
        for cell in ['TP', 'FN', 'FP', 'TN']:
            examples = matrix[cell]['examples']
            if examples:
                cell_full = {
                    'TP': "True Positives",
                    'FN': "False Negatives",
                    'FP': "False Positives",
                    'TN': "True Negatives"
                }[cell]
                print(f"Examples for {cell_full}:")
                for ex in examples:
                    print(f"  PatientID: {ex['patientid']}, Outcome: {ex['outcome']}, Prediction: {ex['prediction']}")
                    print(f"    Narrative: {ex['narrative_text']}")
                print()

def main():
    parser = argparse.ArgumentParser(description="Show confusion matrix for a round")
    parser.add_argument('--database', default='titanic_medical.sqlite', help="Path to the SQLite database file")
    parser.add_argument('--round-id', type=int, required=True, help="Round ID")
    parser.add_argument('--example-count', type=int, default=3, help="Number of examples per cell")
    parser.add_argument('--show-history', type=int, default=2, help="Show confusion matrices for the previous N rounds")
    args = parser.parse_args()

    try:
        conn = sqlite3.connect(args.database)
    except Exception as e:
        print(f"Failed to connect to database '{args.database}': {e}", file=sys.stderr)
        sys.exit(1)

    # Current round.
    prompt = get_round_prompt(conn, args.round_id)
    if prompt is None:
        print(f"Round ID {args.round_id} not found.", file=sys.stderr)
        sys.exit(1)

    matrix = get_confusion_matrix(conn, args.round_id, example_count=args.example_count)
    print("Current Round:")
    print_confusion_matrix(args.round_id, prompt, matrix, show_examples=True)

    # History rounds (without examples)
    if args.show_history > 0:
        cur = conn.cursor()
        cur.execute("""
            SELECT round_id, prompt 
              FROM rounds 
             WHERE round_id < ? 
          ORDER BY round_id DESC 
             LIMIT ?
        """, (args.round_id, args.show_history))
        history_rounds = cur.fetchall()

        if history_rounds:
            print("History:")
            for r_id, r_prompt in history_rounds:
                hist_matrix = get_confusion_matrix(conn, r_id, example_count=0)
                print_confusion_matrix(r_id, r_prompt, hist_matrix, show_examples=False)
        else:
            print("No previous rounds found.")

if __name__ == '__main__':
    main()
