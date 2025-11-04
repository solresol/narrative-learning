#!/usr/bin/env python3
"""Tests for standalone.py DataPainter integration."""

import unittest
import sqlite3
import tempfile
from pathlib import Path
from standalone import load_dataset, DatasetRow


class TestDataPainterLoading(unittest.TestCase):
    """Test loading DataPainter SQLite database files."""

    def setUp(self):
        """Create a temporary DataPainter database for testing."""
        self.temp_db = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sqlite')
        self.temp_db.close()
        self.db_path = Path(self.temp_db.name)

        # Recreate the database from the SQL dump
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create the schema and load test data
        sql_dump_path = Path(__file__).parent / "tests" / "fixtures" / "hamsters.sql"
        with open(sql_dump_path, 'r') as f:
            sql_script = f.read()
            cursor.executescript(sql_script)

        conn.commit()
        conn.close()

    def tearDown(self):
        """Clean up temporary database."""
        if self.db_path.exists():
            self.db_path.unlink()

    def test_load_datapainter_default_table(self):
        """Test loading a DataPainter file without specifying table name."""
        rows = load_dataset(self.db_path)

        # Should load the first table in metadata (hamsters)
        self.assertGreater(len(rows), 0)
        self.assertIsInstance(rows[0], DatasetRow)

        # Check that we have the expected structure
        first_row = rows[0]
        self.assertIsInstance(first_row.feature_a, str)
        self.assertIsInstance(first_row.feature_b, str)
        self.assertIsInstance(first_row.label, str)

        # Check that labels are from the target column
        labels = {row.label for row in rows}
        self.assertEqual(labels, {"male", "female"})

    def test_load_datapainter_explicit_table(self):
        """Test loading a DataPainter file with explicit table name."""
        rows = load_dataset(self.db_path, table_name="hamsters")

        self.assertGreater(len(rows), 0)
        self.assertEqual(len(rows), 31)  # From the sample database

    def test_load_datapainter_invalid_table(self):
        """Test that requesting a non-existent table raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            load_dataset(self.db_path, table_name="nonexistent")

        self.assertIn("not found", str(ctx.exception))
        self.assertIn("hamsters", str(ctx.exception))

    def test_load_datapainter_non_datapainter_file(self):
        """Test that loading a non-DataPainter SQLite file raises ValueError."""
        # Create a SQLite file without metadata table
        temp_db = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sqlite')
        temp_db.close()
        temp_path = Path(temp_db.name)

        try:
            conn = sqlite3.connect(temp_path)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE some_table (id INTEGER PRIMARY KEY)")
            conn.commit()
            conn.close()

            with self.assertRaises(ValueError) as ctx:
                load_dataset(temp_path)

            self.assertIn("not appear to be a DataPainter file", str(ctx.exception))
        finally:
            temp_path.unlink()

    def test_load_dataset_with_table(self):
        """Test that load_dataset accepts table_name parameter."""
        rows = load_dataset(self.db_path, table_name="hamsters")

        self.assertEqual(len(rows), 31)

    def test_datapainter_coordinates_as_strings(self):
        """Test that x/y coordinates are properly converted to strings."""
        rows = load_dataset(self.db_path)

        # All features should be strings (even though they're REAL in DB)
        for row in rows:
            self.assertIsInstance(row.feature_a, str)
            self.assertIsInstance(row.feature_b, str)

            # Should be convertible back to float
            float(row.feature_a)
            float(row.feature_b)


class TestDataPainterMetadata(unittest.TestCase):
    """Test metadata handling for DataPainter files."""

    def setUp(self):
        """Create a temporary DataPainter database with metadata."""
        self.temp_db = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sqlite')
        self.temp_db.close()
        self.db_path = Path(self.temp_db.name)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create metadata table
        cursor.execute("""
            CREATE TABLE metadata (
                table_name        TEXT PRIMARY KEY,
                x_axis_name       TEXT NOT NULL,
                y_axis_name       TEXT NOT NULL,
                target_col_name   TEXT NOT NULL,
                x_meaning         TEXT NOT NULL,
                o_meaning         TEXT NOT NULL,
                valid_x_min       REAL,
                valid_x_max       REAL,
                valid_y_min       REAL,
                valid_y_max       REAL,
                show_zero_bars    INTEGER NOT NULL DEFAULT 0
            )
        """)

        # Insert metadata for two tables
        cursor.execute("""
            INSERT INTO metadata VALUES
            ('dataset1', 'x', 'y', 'label', 'meaning_x', 'meaning_o', 0, 10, 0, 10, 0)
        """)

        cursor.execute("""
            INSERT INTO metadata VALUES
            ('dataset2', 'a', 'b', 'class', 'meaning_a', 'meaning_b', 0, 5, 0, 5, 0)
        """)

        # Create data tables
        cursor.execute("""
            CREATE TABLE dataset1 (
                id INTEGER PRIMARY KEY,
                x REAL NOT NULL,
                y REAL NOT NULL,
                target TEXT NOT NULL
            )
        """)

        cursor.execute("""
            INSERT INTO dataset1 (x, y, target) VALUES
            (1.0, 2.0, 'A'),
            (3.0, 4.0, 'B')
        """)

        cursor.execute("""
            CREATE TABLE dataset2 (
                id INTEGER PRIMARY KEY,
                x REAL NOT NULL,
                y REAL NOT NULL,
                target TEXT NOT NULL
            )
        """)

        cursor.execute("""
            INSERT INTO dataset2 (x, y, target) VALUES
            (0.5, 1.5, 'X'),
            (2.5, 3.5, 'Y'),
            (4.5, 4.5, 'Z')
        """)

        conn.commit()
        conn.close()

    def tearDown(self):
        """Clean up temporary database."""
        if self.db_path.exists():
            self.db_path.unlink()

    def test_load_first_table_by_default(self):
        """Test that without table_name, the first table in metadata is loaded."""
        rows = load_dataset(self.db_path)

        # Should load dataset1 (first in metadata)
        self.assertEqual(len(rows), 2)
        labels = {row.label for row in rows}
        self.assertEqual(labels, {"A", "B"})

    def test_load_second_table_explicitly(self):
        """Test loading a specific table when multiple are available."""
        rows = load_dataset(self.db_path, table_name="dataset2")

        self.assertEqual(len(rows), 3)
        labels = {row.label for row in rows}
        self.assertEqual(labels, {"X", "Y", "Z"})

    def test_empty_metadata_table(self):
        """Test that empty metadata table raises ValueError."""
        temp_db = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sqlite')
        temp_db.close()
        temp_path = Path(temp_db.name)

        try:
            conn = sqlite3.connect(temp_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE metadata (
                    table_name TEXT PRIMARY KEY
                )
            """)
            conn.commit()
            conn.close()

            with self.assertRaises(ValueError) as ctx:
                load_dataset(temp_path)

            self.assertIn("No tables found in metadata", str(ctx.exception))
        finally:
            temp_path.unlink()


if __name__ == '__main__':
    unittest.main()
