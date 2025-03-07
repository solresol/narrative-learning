#!/usr/bin/env python3
import sqlite3
import sys
import json
import os
from typing import Dict, List, Any, Optional, Tuple

class TargetClassingException(Exception):
    def __init__(self, target, num_classes):
        self.message = f"{target} has {num_classes}, but narrative learning can only cope with binary classification at the moment"
        super().__init__(self.message)

    def __str__(self):
        return self.message

class MissingConfigElementException(Exception):
    """Exception raised when a required configuration element is missing.

    Attributes:
        column_name -- The name of the missing configuration element
        config_path -- The path to the configuration file being read
    """

    def __init__(self, column_name, config_path):
        self.column_name = column_name
        self.config_path = config_path
        self.message = f"Missing required configuration element: '{column_name}' in config file: '{config_path}'"
        super().__init__(self.message)

    def __str__(self):
        return self.message

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
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Required configuration fields
        required_fields = [ "table_name", "primary_key", "target_field", "splits_table", "columns" ]

        # Verify configuration has all required fields
        for field in required_fields:
            if field not in self.config:
                raise MissingConfigElementException(field, config_path)

        # Store key configuration values as attributes for convenience
        self.table_name = self.config["table_name"]
        self.primary_key = self.config["primary_key"]
        self.target_field = self.config["target_field"]
        self.splits_table = self.config["splits_table"]
        self.columns = self.config["columns"]

        cursor = conn.cursor()
        cursor.execute(f"select distinct {self.target_field} from {self.table_name}")
        self.valid_predictions = []
        for row in cursor:
            self.valid_predictions.append(row[0])
        if len(self.valid_predictions) != 2:
            raise TargetClassingException(self.target_field, len(self.valid_predictions))

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
            if column != self.primary_key and column != self.target_field:
                columns_to_select.append(column)

        column_list = ", ".join([f'"{col}"' for col in columns_to_select])

        query = f"""
        SELECT {column_list}
        FROM {self.table_name}
        WHERE {self.primary_key} = ?
        """

        cur = self.conn.cursor()
        cur.execute(query, (entity_id,))
        row = cur.fetchone()

        if not row:
            raise KeyError(f"Entity ID '{entity_id}' not found.")

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
        # I think this function isn't used. The only difference between
        # this and get_entity_features is that this supplies the entity_id
        # (which the caller already knew) and the target (which is only used
        # in the training phase, when it's an aggregate query and we wouldn't
        # ask for one element).
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
        FROM {self.splits_table}
        WHERE {self.primary_key} = ? AND split_id = ?
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
            SELECT m.{self.target_field}, i.prediction, i.{self.primary_key},
                   i.narrative_text, s.holdout, s.validation
              FROM inferences i
              JOIN {self.table_name} m ON i.{self.primary_key} = m.{self.primary_key}
              JOIN {self.entity_split_table} s ON (s.{self.primary_key} = i.{self.primary_key})
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
