#!/usr/bin/env python3
import sqlite3
import sys
import json
import os
import pandas as pd
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
        cursor.execute(f"select distinct {self.target_field} from {self.table_name} order by {self.target_field}")
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

    def get_random_non_holdout_id(self, split_id: int) -> str:
        """
        Retrieve the ID of a random entity that is not part of the holdout dataset.

        Args:
           split_id: The split ID to check against

        Returns:
           A random entity ID that is not in the holdout set

        Raises:
           ValueError: If no non-holdout data is available for the given split
        """
        cur = self.conn.cursor()
        query = f"""
           SELECT s.{self.primary_key}
           FROM {self.splits_table} s
           WHERE s.split_id = ? AND s.holdout = 0
           ORDER BY RANDOM()
           LIMIT 1
        """

        cur.execute(query, (split_id,))
        row = cur.fetchone()

        if row is None:
            raise ValueError(f"No non-holdout data available for split ID {split_id}")

        return row[0]

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

    def positive_label(self):
        # This ain't great. Fundamentally, we are trying to shoe-horn into a "true positive" / "false positive"
        # labelling system and it's not all that well-defined what positive or negative is here.
        return self.valid_predictions[0]

    def negative_label(self):
        return self.valid_predictions[1]

    def get_matrix_label_for_prediction(self, ground_truth, prediction):
        if ground_truth == self.positive_label():
            if ground_truth == prediction:
                return 'TP'
            return 'FN'
        if ground_truth == prediction:
            return 'TN'
        else:
            return 'FP'

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
              JOIN {self.splits_table} s ON (s.{self.primary_key} = i.{self.primary_key})
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

            cell = self.get_matrix_label_for_prediction(outcome, prediction)

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
                    answer += (f"Examples for {cell_full}: (Correct answer: {ex['outcome']}, What the previous set of rules predicted: {ex['prediction']})\n")
                    for ex in examples:
                        answer += (f"  Entity Data:\n{ex['features']}\n")
                    answer += "\n"

        return answer


    def calculate_metric(self, matrix: Dict, metric_name: str) -> float:
        """
        Calculate the specified metric from a confusion matrix.

        Args:
            matrix: Confusion matrix dictionary
            metric_name: Name of the metric to calculate ('count', 'accuracy', 'precision', 'recall', 'f1')

        Returns:
            Float value of the calculated metric

        Raises:
            ValueError: If the metric name is unknown
        """
        tp = matrix['TP']['count']
        fn = matrix['FN']['count']
        fp = matrix['FP']['count']
        tn = matrix['TN']['count']

        if metric_name == 'count':
            return tp + fn + fp + tn
        if metric_name == 'accuracy':
            total = tp + fn + fp + tn
            return (tp + tn) / total if total > 0 else 0
        elif metric_name == 'precision':
            return tp / (tp + fp) if (tp + fp) > 0 else 0
        elif metric_name == 'recall':
            return tp / (tp + fn) if (tp + fn) > 0 else 0
        elif metric_name == 'f1':
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    def get_latest_split_id(self) -> int:
        """
        Get the split_id from the most recent round.

        Returns:
            Integer split ID

        Raises:
            SystemExit: If no rounds are found in the database
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT split_id
            FROM rounds
            ORDER BY round_id DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        if row is None:
            sys.exit("No rounds found in database")
        split_id = row[0]
        return split_id

    def get_rounds_for_split(self, split_id: int) -> List[int]:
        """
        Get all round IDs for a given split_id.

        Args:
            split_id: The split ID to query

        Returns:
            List of round IDs
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT round_id
            FROM rounds
            WHERE split_id = ?
            ORDER BY round_id
        """, (split_id,))
        rounds = [row[0] for row in cur.fetchall()]
        return rounds

    def get_processed_rounds_for_split(self, split_id: int) -> List[int]:
        """
        Get all round IDs for a given split_id that have inferences.

        Args:
            split_id: The split ID to query

        Returns:
            List of round IDs that have inferences
        """
        cur = self.conn.cursor()
        answer = []
        for r in self.get_rounds_for_split(split_id):
            cur.execute("select count(*) from inferences where round_id = ?", [r])
            row = cur.fetchone()
            if row[0] == 0:
                continue
            answer.append(r)
        return answer

    def check_early_stopping(self, split_id: int, metric: str,
                            patience: int, on_validation: bool = True) -> bool:
        """
        Check if training should be stopped based on validation performance.

        Args:
            split_id: The split ID to check
            metric: The metric to use for evaluation ('accuracy', 'precision', 'recall', 'f1')
            patience: Number of rounds to look back for improvement
            on_validation: Whether to use validation data

        Returns:
            True if training should stop, False otherwise
        """
        rounds = self.get_processed_rounds_for_split(split_id)
        if len(rounds) < patience + 1:
            return False

        # Look at last 'patience' + 1 rounds
        relevant_rounds = rounds[-(patience + 1):]
        oldest_round = relevant_rounds[0]

        # Calculate metric for oldest round
        oldest_matrix = self.get_confusion_matrix(oldest_round, on_holdout_data=on_validation)
        best_score = self.calculate_metric(oldest_matrix, metric)

        # Check if any later round beat this score
        for round_id in relevant_rounds[1:]:
            matrix = self.get_confusion_matrix(round_id, on_holdout_data=on_validation)
            score = self.calculate_metric(matrix, metric)
            if score > best_score:
                return False

        return True

    def generate_metrics_data(self, split_id: int, metric: str, data_type: str) -> pd.DataFrame:
        """
        Generate a DataFrame with metrics for all rounds in a split.

        Args:
            split_id: The split ID to analyze
            metric: The metric to calculate ('accuracy', 'precision', 'recall', 'f1')
            data_type: Type of data to analyze ('train', 'validation', or 'test')

        Returns:
            DataFrame with round_id and metric columns
        """
        rounds = self.get_processed_rounds_for_split(split_id)
        data = []

        for round_id in rounds:
            # Set appropriate flags based on data_type
            on_holdout = data_type in ('validation', 'test')
            on_test_data = data_type == 'test'
            matrix = self.get_confusion_matrix(round_id, on_holdout_data=on_holdout, on_test_data=on_test_data)
            score = self.calculate_metric(matrix, metric)
            data.append({
                'round_id': round_id,
                'metric': score
            })
            
        return pd.DataFrame(data)
