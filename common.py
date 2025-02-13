#!/usr/bin/env python3
import sqlite3
import sys

def get_round_prompt(conn, round_id):
    """
    Return the prompt text for a given round.
    """
    cur = conn.cursor()
    cur.execute("SELECT prompt FROM rounds WHERE round_id = ?", (round_id,))
    row = cur.fetchone()
    if row is None:
        sys.exit(f"Asked to get prompt from {round_id=}")
    return row[0]


def get_split_id(conn, round_id):
    cur = conn.cursor()
    cur.execute("select split_id from rounds where round_id = ?", [round_id])
    row = cur.fetchone()
    if row is None:
        sys.exit(f"Asked to get split ID from {round_id=}")
    split_id = row[0]
    return split_id

def is_holdout_data(conn, patient_id, split_id):
    cur = conn.cursor()
    cur.execute("select holdout from patient_split where patient_id = ?", [patient_id])
    row = cur.fetchone()
    if row is None:
        raise KeyError(patient_id)
    return row[0]

def get_confusion_matrix(conn, round_id, example_count=0, on_holdout_data=False, on_test_data=False):
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
    split_id = get_split_id(conn, round_id)
    cur.execute("""
        SELECT m.Outcome, i.prediction, i.patient_id, i.narrative_text, holdout, validation
          FROM inferences i
          JOIN medical_treatment_data m ON i.patient_id = m.Patient_ID
          JOIN patient_split using (patient_id)
         WHERE i.round_id = ? and i.patient_id and split_id = ?
         order by random()
    """, (round_id,split_id))
    rows = cur.fetchall()

    # Initialise confusion matrix cells.
    matrix = {
        'TP': {'count': 0, 'examples': []},
        'FN': {'count': 0, 'examples': []},
        'FP': {'count': 0, 'examples': []},
        'TN': {'count': 0, 'examples': []},
    }

    for outcome, prediction, patientid, narrative_text, holdout, validation in rows:
        if on_holdout_data and not holdout:
            continue
        if not on_holdout_data and holdout:
            continue
        if on_holdout_data and holdout:
            if validation and on_test_data:
                continue
            if not validation and not on_test_data:
                continue
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
                'features': get_patient_features(conn, patientid),
                'narrative_text': narrative_text.replace('\n', '\n\t'),
                'outcome': outcome,
                'prediction': prediction
            })
    return matrix


def get_patient_features(conn, patient_id):
    # Retrieve patient data, excluding PatientID, Decodex, and Outcome.
    patient_query = """
    SELECT Treatment_Group, Sex, Treatment_Months, Genetic_Class_A_Matches,
           Genetic_Class_B_Matches, TcQ_mass, Cohort
      FROM medical_treatment_data
     WHERE Patient_ID = ?
    """
    cur = conn.cursor()
    cur.execute(patient_query, (patient_id,))
    patient_row = cur.fetchone()
    if not patient_row:
        sys.exit(f"Patient ID '{patient_id}' not found.")

    columns = [
        "Treatment Group", "Sex", "Treatment Months",
        "Genetic_Class_A_Matches", "Genetic_Class_B_Matches",
        "TcQ_mass", "Cohort"
    ]
    answer = ""
    for col, value in zip(columns, patient_row):
        answer += f"\t{col}: {value}\n"
    return answer


def get_printable_confusion_matrix_and_examples(round_id, prompt, matrix, show_examples=True):
    answer = ""
    answer += f"Round ID: {round_id}\n"
    answer += "Prompt used:\n\t"
    answer += prompt.replace('\n', '\n\t')
    answer += "\n\nConfusion Matrix:\n"
    # Layout: rows are Actual values; columns are Predicted.
    answer += (f"{'':15s} {'Predicted Positive':20s} {'Predicted Negative':20s}\n")
    # For actual positive:
    tp = matrix['TP']['count']
    fn = matrix['FN']['count']
    answer += (f"{'Actual Positive':15s} {tp:20d} {fn:20d}\n")
    # For actual negative:
    fp = matrix['FP']['count']
    tn = matrix['TN']['count']
    answer += (f"{'Actual Negative':15s} {fp:20d} {tn:20d}\n")
    answer += "\n"
    total_count = tp + fn + fp + tn
    accuracy = (tp + tn) / total_count
    answer += f"Accuracy: {accuracy:.3f}\n"
    # Calculate precision, recall, and F1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    answer += f"Precision: {precision:.3f}\n"
    answer += f"Recall: {recall:.3f}\n"
    answer += f"F1 Score: {f1_score:.3f}\n\n"
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
                ex = examples[0]
                answer += (f"Examples for {cell_full}: (Outcome: {ex['outcome']}, Prediction: {ex['prediction']})\n")
                for ex in examples:
                    #answer += (f"  PatientID: {ex['patientid']}, 
                    answer += (f"  PatientData:\n{ex['features']}\n")
                    #answer += (f"    Narrative: {ex['narrative_text']}")
                answer += "\n"
    return answer

