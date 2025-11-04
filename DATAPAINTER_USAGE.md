# Using DataPainter Files with standalone.py

## Overview
The `standalone.py` tool now supports loading datasets from DataPainter SQLite database files in addition to CSV files.

## Quick Start

### Loading a DataPainter File
```bash
# Auto-detect and load first table
uv run standalone.py --dataset /path/to/file.sqlite

# Specify a specific table
uv run standalone.py --dataset /path/to/file.sqlite --table tablename
```

### Example with Sample File
```bash
# Load the hamsters dataset
uv run standalone.py --dataset /tmp/1.sqlite --table hamsters
```

## How It Works

### File Format Detection
The tool automatically detects the file format based on file extension:
- `.csv` → CSV format (feature_a, feature_b, label columns)
- `.sqlite`, `.sqlite3`, `.db` → DataPainter SQLite format

### DataPainter Format
DataPainter files contain:
1. **metadata table** - Configuration for each dataset
   - table names, axis names, valid ranges, etc.
2. **data tables** - Actual data points
   - `id` (primary key)
   - `x` (feature_a)
   - `y` (feature_b)
   - `target` (label)

### Table Selection
- If `--table` is not specified, the first table in the metadata is used
- If `--table` is specified, that specific table is loaded
- If the specified table doesn't exist, you'll get an error listing available tables

## Testing

Run the test suite to verify DataPainter support:
```bash
uv run python -m unittest test_standalone -v
```

## Command-Line Options

```
--dataset PATH        Path to dataset file (CSV or DataPainter SQLite)
--table NAME          Table name for DataPainter files (optional)
--database PATH       Path for results database (defaults to dataset.sqlite3)
--export-json PATH    Export results to JSON file
--shuffle-seed INT    Random seed for train/val split (default: 13)
--max-rounds INT      Maximum number of training rounds
```

## Examples

### List Available Tables
To see what tables are available in a DataPainter file:
```bash
sqlite3 file.sqlite "SELECT table_name FROM metadata"
```

### Inspect Table Metadata
```bash
sqlite3 file.sqlite "SELECT * FROM metadata WHERE table_name='tablename'"
```

### View Sample Data
```bash
sqlite3 file.sqlite "SELECT * FROM tablename LIMIT 10"
```

## Programmatic Usage

```python
from pathlib import Path
from standalone import load_dataset

# Auto-detect format and load
rows = load_dataset(Path("/tmp/1.sqlite"))

# Specify table explicitly
rows = load_dataset(Path("/tmp/1.sqlite"), table_name="hamsters")

# Access data
for row in rows[:5]:
    print(f"({row.feature_a}, {row.feature_b}) -> {row.label}")
```

## Data Conversion

DataPainter stores coordinates as `REAL` (floating-point) values, but standalone.py converts them to strings to maintain compatibility with the CSV format. The conversion is transparent and maintains precision:

```
Database: x=4.55696, y=6.95652, target="male"
↓
DatasetRow: feature_a="4.55696", feature_b="6.95652", label="male"
```

## Troubleshooting

### "Does not appear to be a DataPainter file"
The SQLite file is missing the required `metadata` table. Ensure you're using a file created by DataPainter.

### "Table 'X' not found"
The specified table name doesn't exist in the metadata. Check available tables with:
```bash
sqlite3 file.sqlite "SELECT table_name FROM metadata"
```

### "Dataset is empty"
The specified table exists but contains no rows. Check with:
```bash
sqlite3 file.sqlite "SELECT COUNT(*) FROM tablename"
```

## See Also
- `DATAPAINTER_FORMAT.md` - Detailed database schema documentation
- `test_standalone.py` - Test suite with usage examples
- `tests/fixtures/hamsters.sql` - Sample database for testing
