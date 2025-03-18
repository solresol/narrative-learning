#!/usr/bin/env python3
import unittest
import os
import tempfile
from env_settings import EnvSettings

class TestEnvSettings(unittest.TestCase):
    
    def test_from_file(self):
        # Create a temporary env file
        content = """
        NARRATIVE_LEARNING_DATABASE=/path/to/db.sqlite
        NARRATIVE_LEARNING_CONFIG=/path/to/config.json
        NARRATIVE_LEARNING_TRAINING_MODEL=gpt-4o
        NARRATIVE_LEARNING_EXAMPLE_COUNT=10
        """
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
            temp.write(content)
            temp_path = temp.name
        
        try:
            # Test parsing
            settings = EnvSettings.from_file(temp_path)
            
            # Check values
            self.assertEqual(settings.database, '/path/to/db.sqlite')
            self.assertEqual(settings.config, '/path/to/config.json')
            self.assertEqual(settings.model, 'gpt-4o')
            self.assertEqual(settings.sampler, 10)
            
            # Test validation
            self.assertTrue(settings.is_valid())
            valid, _ = settings.validate()
            self.assertFalse(valid)  # Files don't exist
            
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_invalid_settings(self):
        # Test with empty settings
        settings = EnvSettings()
        self.assertFalse(settings.is_valid())
        
        # Test with partial settings
        settings = EnvSettings(database='/path/to/db.sqlite')
        self.assertFalse(settings.is_valid())
        
        # Test with complete settings
        settings = EnvSettings(
            database='/path/to/db.sqlite',
            config='/path/to/config.json',
            model='gpt-4o'
        )
        self.assertTrue(settings.is_valid())

if __name__ == '__main__':
    unittest.main()