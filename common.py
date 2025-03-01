#!/usr/bin/env python3
import sqlite3
import sys
import json
import os
from typing import Dict, List, Any, Optional, Tuple

class DatasetConfig:
    """
    Class for handling dataset configuration and database connection.
    This class provides methods for retrieving and handling obfuscated data.
    """

    def __init__(self, conn: sqlite3.Connection, config_path: str):
        """
        Initialize with a database connection and configuration file path.

        Args:
            conn: SQLite database connection
            config_path: Path to the JSON configuration file
        """
        self.conn = conn

        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            sys.exit(f"Failed to load configuration from {config_path}: {e}")

        # Required configuration fields
        required_fields = [
            "table_name", "primary_key_field", "target_field",
            "entity_id_field", "entity_split_table", "columns"
        ]

        # Verify configuration has all required fields
        for field in required_fields:
            if field not in self.config:
                sys.exit(f"Configuration missing required field: {field}")

        # Store key configuration values as attributes for convenience
        self.table_name = self.config["table_name"]
        self.primary_key_field = self.config["primary_key_field"]
        self.target_field = self.config["target_field"]
        self.entity_id_field = self.config["entity_id_field"]
        self.entity_split_table = self.config["entity_split_table"]
        self.columns = self.config["columns"]

    def get_entity_features(self, entity_id: str) -> str:
        """
        Retrieve the features for a specific entity as a formatted string.

        Args:
            entity_id: The ID of the entity to retrieve

        Returns:
            Formatted string with entity features
        """
        # Build a query that selects all columns except primary key and target
        columns_to_select = []
        for column in self.columns:
            if column != self.primary_key_field and column != self.target_field:
                columns_to_select.append(column)

        column_list = ", ".join([f'"{col}"' for col in columns_to_select])

        query = f"""
        SELECT {column_list}
        FROM {self.table_name}
        WHERE {self.primary_key_field} = ?
        """

        cur = self.conn.cursor()
        cur.execute(query, (entity_id,))
        row = cur.fetchone()

        if not row:
            sys.exit(f"Entity ID '{entity_id}' not found.")

        # Format the result as a string
        result = ""
        for i, col in enumerate(columns_to_select):
            result += f"\t{col}: {row[i]}\n"

        return result

    def get_entity_by_id(self, entity_id: str) -> Dict[str, Any]:
        """
        Retrieve all data for a specific entity.

        Args:
            entity_id: The ID of the entity to retrieve

        Returns:
            Dictionary with entity data
        """
        column_list = ", ".join([f'"{col}"' for col in self.columns])

        query = f"""
        SELECT {column_list}
        FROM {self.table_name}
        WHERE {self.primary_key_field} = ?
        """

        cur = self.conn.cursor()
        cur.execute(query, (entity_id,))
        row = cur.fetchone()

        if not row:
            sys.exit(f"Entity ID '{entity_id}' not found.")

        # Convert to dictionary
        result = {}
        for i, col in enumerate(self.columns):
            result[col] = row[i]

        return result

    def is_holdout_data(self, entity_id: str, split_id: int) -> bool:
        """
        Check if an entity is part of the holdout dataset.

        Args:
            entity_id: The ID of the entity to check
            split_id: The split ID to check against

        Returns:
            True if the entity is holdout data, False otherwise
        """
        cur = self.conn.cursor()
        query = f"""
        SELECT holdout
        FROM {self.entity_split_table}
        WHERE {self.entity_id_field}_id = ? AND split_id = ?
        """

        cur.execute(query, (entity_id, split_id))
        row = cur.fetchone()

        if row is None:
            raise KeyError(entity_id)

        return bool(row[0])

    def get_round_prompt(self, round_id: int) -> str:
        """
        Retrieve the prompt text for a given round.

        Args:
            round_id: ID of the round

        Returns:
            Prompt text
        """
        cur = self.conn.cursor()
        cur.execute("SELECT prompt FROM rounds WHERE round_id = ?", (round_id,))
        row = cur.fetchone()

        if row is None:
            sys.exit(f"Round ID {round_id} not found")

        return row[0]

    def get_split_id(self, round_id: int) -> int:
        """
        Get the split ID associated with a round.

        Args:
            round_id: ID of the round

        Returns:
            Split ID
        """
        cur = self.conn.cursor()
        cur.execute("SELECT split_id FROM rounds WHERE round_id = ?", (round_id,))
        row = cur.fetchone()

        if row is None:
            sys.exit(f"Round ID {round_id} not found")

        return row[0]

    def get_confusion_matrix(self, round_id: int, example_count: int = 0,
                           on_holdout_data: bool = False, on_test_data: bool = False) -> Dict:
        """
        Retrieve the confusion matrix for a given round.

        Args:
            round_id: ID of the round
            example_count: Number of examples to include per cell (0 for none)
            on_holdout_data: Whether to use holdout data
            on_test_data: Whether to use test data (subset of holdout)

        Returns:
            Confusion matrix dictionary
        """
        cur = self.conn.cursor()
        split_id = self.get_split_id(round_id)

        query = f"""
            SELECT m.{self.target_field}, i.prediction, i.{self.entity_id_field}_id,
                   i.narrative_text, s.holdout, s.validation
              FROM inferences i
              JOIN {self.table_name} m ON i.{self.entity_id_field}_id = m.{self.primary_key_field}
              JOIN {self.entity_split_table} s ON (s.{self.entity_id_field}_id = i.{self.entity_id_field}_id)
             WHERE i.round_id = ? AND s.split_id = ?
             ORDER BY RANDOM()
        """

        cur.execute(query, (round_id, split_id))
        rows = cur.fetchall()

        # Initialize confusion matrix cells
        matrix = {
            'TP': {'count': 0, 'examples': []},
            'FN': {'count': 0, 'examples': []},
            'FP': {'count': 0, 'examples': []},
            'TN': {'count': 0, 'examples': []},
        }

        for outcome, prediction, entity_id, narrative_text, holdout, validation in rows:
            if on_holdout_data and not holdout:
                continue
            if not on_holdout_data and holdout:
                continue
            if on_holdout_data and holdout:
                if validation and on_test_data:
                    continue
                if not validation and not on_test_data:
                    continue

            # Standardize outcome and prediction values
            outcome_label = 'Success' if outcome.strip().lower() == 'success' else 'Failure'
            prediction_label = 'Success' if prediction.strip().lower() == 'success' else 'Failure'

            # Determine which matrix cell this belongs to
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

            # Update counts and collect examples if needed
            matrix[cell]['count'] += 1
            if example_count > 0 and len(matrix[cell]['examples']) < example_count:
                matrix[cell]['examples'].append({
                    'entity_id': entity_id,
                    'features': self.get_entity_features(entity_id),
                    'narrative_text': narrative_text.replace('\n', '\n\t'),
                    'outcome': outcome,
                    'prediction': prediction
                })

        return matrix

    def get_printable_confusion_matrix_and_examples(self, round_id: int, matrix: Dict, show_examples: bool = True) -> str:
        """
        Format a confusion matrix as a printable string.

        Args:
            round_id: ID of the round
            matrix: Confusion matrix from get_confusion_matrix
            show_examples: Whether to include examples in the output

        Returns:
            Formatted string representation
        """
        prompt = self.get_round_prompt(round_id)

        answer = ""
        answer += f"Round ID: {round_id}\n"
        answer += "Prompt used:\n\t"
        answer += prompt.replace('\n', '\n\t')
        answer += "\n\nConfusion Matrix:\n"

        # Layout: rows are Actual values; columns are Predicted
        answer += (f"{'':15s} {'Predicted Positive':20s} {'Predicted Negative':20s}\n")

        # For actual positive
        tp = matrix['TP']['count']
        fn = matrix['FN']['count']
        answer += (f"{'Actual Positive':15s} {tp:20d} {fn:20d}\n")

        # For actual negative
        fp = matrix['FP']['count']
        tn = matrix['TN']['count']
        answer += (f"{'Actual Negative':15s} {fp:20d} {tn:20d}\n")
        answer += "\n"

        # Calculate metrics
        total_count = tp + fn + fp + tn
        accuracy = (tp + tn) / total_count if total_count > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Add metrics to output
        answer += f"Accuracy: {accuracy:.3f}\n"
        answer += f"Precision: {precision:.3f}\n"
        answer += f"Recall: {recall:.3f}\n"
        answer += f"F1 Score: {f1_score:.3f}\n\n"

        # Add examples if requested
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
                        answer += (f"  Entity Data:\n{ex['features']}\n")
                    answer += "\n"

        return answer

# Helper function to find and load config file
def find_config_for_database(database_path):
    """
    Find the corresponding config file for a database.

    Args:
        database_path: Path to the database file

    Returns:
        Path to the config file
    """
    # Check for config file in the same directory with .config.json extension
    config_path = os.path.splitext(database_path)[0] + '.config.json'
    if os.path.exists(config_path):
        return config_path

    # If not found, look for any .config.json file in the same directory
    directory = os.path.dirname(database_path)
    for file in os.listdir(directory):
        if file.endswith('.config.json'):
            return os.path.join(directory, file)

    sys.exit(f"Could not find a config file for database {database_path}")

# Function to create a DatasetConfig from a database path
def create_config_from_database(database_path, config_path=None):
    """
    Create a DatasetConfig object from a database path.

    Args:
        database_path: Path to the database file
        config_path: Optional path to the config file

    Returns:
        DatasetConfig object
    """
    try:
        conn = sqlite3.connect(database_path)
    except Exception as e:
        sys.exit(f"Failed to connect to database '{database_path}': {e}")

    if config_path is None:
        config_path = find_config_for_database(database_path)

    return DatasetConfig(conn, config_path)

# These functions remain for backward compatibility
def get_round_prompt(conn, round_id):
    """Legacy function that creates a temporary DatasetConfig to get round prompt."""
    # This is inefficient but maintains backward compatibility
    cursor = conn.cursor()
    cursor.execute("PRAGMA database_list")
    db_path = cursor.fetchone()[2]  # Get the database file path

    config_path = find_config_for_database(db_path)
    dataset_config = DatasetConfig(conn, config_path)
    return dataset_config.get_round_prompt(round_id)

def get_split_id(conn, round_id):
    """Legacy function that creates a temporary DatasetConfig to get split ID."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA database_list")
    db_path = cursor.fetchone()[2]  # Get the database file path

    config_path = find_config_for_database(db_path)
    dataset_config = DatasetConfig(conn, config_path)
    return dataset_config.get_split_id(round_id)

def is_holdout_data(conn, entity_id, split_id):
    """Legacy function that creates a temporary DatasetConfig to check holdout status."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA database_list")
    db_path = cursor.fetchone()[2]  # Get the database file path

    config_path = find_config_for_database(db_path)
    dataset_config = DatasetConfig(conn, config_path)
    return dataset_config.is_holdout_data(entity_id, split_id)

def get_confusion_matrix(conn, round_id, example_count=0, on_holdout_data=False, on_test_data=False):
    """Legacy function that creates a temporary DatasetConfig to get confusion matrix."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA database_list")
    db_path = cursor.fetchone()[2]  # Get the database file path

    config_path = find_config_for_database(db_path)
    dataset_config = DatasetConfig(conn, config_path)
    return dataset_config.get_confusion_matrix(round_id, example_count, on_holdout_data, on_test_data)

def get_patient_features(conn, patient_id):
    """Legacy function for backward compatibility."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA database_list")
    db_path = cursor.fetchone()[2]  # Get the database file path

    config_path = find_config_for_database(db_path)
    dataset_config = DatasetConfig(conn, config_path)
    return dataset_config.get_entity_features(patient_id)

def get_printable_confusion_matrix_and_examples(round_id, prompt, matrix, show_examples=True):
    """Legacy function for backward compatibility."""
    # This doesn't use DatasetConfig but replicates the functionality
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
                    answer += (f"  EntityData:\n{ex['features']}\n")
                answer += "\n"
    return answer
