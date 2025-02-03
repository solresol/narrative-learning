#!/usr/bin/env python3
import sqlite3

def get_round_prompt(conn, round_id):
    """
    Return the prompt text for a given round.
    """
    cur = conn.cursor()
    cur.execute("SELECT prompt FROM rounds WHERE round_id = ?", (round_id,))
    row = cur.fetchone()
    return row[0] if row else None

def get_confusion_matrix(conn, round_id, example_count=0):
    """
    Retrieve the confusion matrix for a given round.
    
    This function joins the inferences with the medical_treatment_data (to get Outcome)
    and then computes counts for each cell:
      - True Positive (TP): Outcome = 'Success' and prediction = 'Success'
      - False Negative (FN): Outcome = 'Success' and prediction = 'Failure'
      - False Positive (FP): Outcome = 'Failure' and prediction = 'Success'
      - True Negative (TN): Outcome = 'Failure' and prediction = 'Failure'
    
    If example_count > 0, up to that many examples (dicts) will be saved per cell.
    Each example is a dict with keys: patientid, narrative_text, outcome, prediction.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT m.Outcome, i.prediction, i.patient_id, i.narrative_text
          FROM inferences i
          JOIN medical_treatment_data m ON i.patient_id = m.Patient_ID
         WHERE i.round_id = ?
    """, (round_id,))
    rows = cur.fetchall()

    # Initialise confusion matrix cells.
    matrix = {
        'TP': {'count': 0, 'examples': []},
        'FN': {'count': 0, 'examples': []},
        'FP': {'count': 0, 'examples': []},
        'TN': {'count': 0, 'examples': []},
    }

    for outcome, prediction, patientid, narrative_text in rows:
        # Assume case-insensitive comparison. If Outcome (or prediction) isn’t exactly 'Success',
        # we treat it as 'Failure'.
        outcome_label = 'Success' if outcome.strip().lower() == 'success' else 'Failure'
        prediction_label = 'Success' if prediction.strip().lower() == 'success' else 'Failure'

        if outcome_label == 'Success' and prediction_label == 'Success':
            cell = 'TP'
        elif outcome_label == 'Success' and prediction_label == 'Failure':
            cell = 'FN'
        elif outcome_label == 'Failure' and prediction_label == 'Success':
            cell = 'FP'
        elif outcome_label == 'Failure' and prediction_label == 'Failure':
            cell = 'TN'
        else:
            continue  # Shouldn't happen

        matrix[cell]['count'] += 1
        if example_count > 0 and len(matrix[cell]['examples']) < example_count:
            matrix[cell]['examples'].append({
                'patientid': patientid,
                'narrative_text': narrative_text,
                'outcome': outcome,
                'prediction': prediction
            })
    return matrix
