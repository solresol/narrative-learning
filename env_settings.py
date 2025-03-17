#!/usr/bin/env python3
import re
import os
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class EnvSettings:
    """Class for holding environment settings from env files."""
    database: Optional[str] = None
    config: Optional[str] = None
    model: Optional[str] = None
    sampler: int = 3
    
    @classmethod
    def from_file(cls, env_file_path: str) -> 'EnvSettings':
        """Create EnvSettings from an env file path."""
        settings = {}
        try:
            with open(env_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract database path
            db_match = re.search(r'NARRATIVE_LEARNING_DATABASE=([^\s]+)', content)
            if db_match:
                settings['database'] = db_match.group(1)

            # Extract config path
            config_match = re.search(r'NARRATIVE_LEARNING_CONFIG=([^\s]+)', content)
            if config_match:
                settings['config'] = config_match.group(1)

            # Extract training model
            model_match = re.search(r'NARRATIVE_LEARNING_TRAINING_MODEL=([^\s]+)', content)
            if model_match:
                settings['model'] = model_match.group(1)

            # Extract example count (sampler)
            example_match = re.search(r'NARRATIVE_LEARNING_EXAMPLE_COUNT=(\d+)', content)
            if example_match:
                settings['sampler'] = int(example_match.group(1))

            return cls(**settings)
        except Exception as e:
            print(f"Error processing env file {env_file_path}: {e}")
            return cls()
    
    def is_valid(self) -> bool:
        """Check if all required settings are present."""
        return all(getattr(self, key) is not None for key in ['database', 'config', 'model'])
    
    def database_exists(self) -> bool:
        """Check if the database file exists."""
        return self.database is not None and os.path.exists(self.database)
    
    def config_exists(self) -> bool:
        """Check if the config file exists."""
        return self.config is not None and os.path.exists(self.config)
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate that all required settings exist and files are accessible.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        if not self.is_valid():
            missing = [key for key in ['database', 'config', 'model'] 
                      if getattr(self, key) is None]
            return False, f"Missing required settings: {', '.join(missing)}"
        
        if not self.database_exists():
            return False, f"Database file not found: {self.database}"
            
        if not self.config_exists():
            return False, f"Config file not found: {self.config}"
            
        return True, None