#!/usr/bin/env python3
from typing import Optional

class TargetClassingException(Exception):
    def __init__(self, target, num_classes):
        self.message = f"{target} has {num_classes}, but narrative learning can only cope with binary classification at the moment"
        super().__init__(self.message)

    def __str__(self):
        return self.message

class NonexistentRoundException(Exception):
    def __init__(self, round_id, db_path=None):
        if db_path is None:
            self.message = f"Round {round_id} not found"
        else:
            self.message = f"Round {round_id} not found in the sqlite database {db_path}"
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


class NoProcessedRoundsException(Exception):
    def __init__(self, split_id, db_path=None):
        if db_path is None:
            self.message = f"No processed rounds found for {split_id}"
        else:
            self.message = f"No processed rounds found for {split_id} in the sqlite database {db_path}"            
        super().__init__(self.message)
    def __str__(self):
        return self.message    
