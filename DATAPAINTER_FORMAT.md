# DataPainter SQLite Database Format

## Overview
DataPainter saves datasets as SQLite database files (.sqlite) with a specific schema that includes metadata and data tables.

## Database Structure

### 1. Metadata Table
The `metadata` table stores information about each data table in the database.

**Schema:**
```sql
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
);
```

**Field Descriptions:**
- `table_name`: Name of the data table (PRIMARY KEY)
- `x_axis_name`: Display name for the X axis
- `y_axis_name`: Display name for the Y axis
- `target_col_name`: Display name for the target/classification column
- `x_meaning`: Semantic meaning/description of X values (e.g., "male")
- `o_meaning`: Semantic meaning/description of O values (e.g., "female")
- `valid_x_min`, `valid_x_max`: Valid range for X coordinates
- `valid_y_min`, `valid_y_max`: Valid range for Y coordinates
- `show_zero_bars`: Boolean flag (0/1) for display settings

**Example Row:**
```
table_name: hamsters
x_axis_name: height
y_axis_name: weight
target_col_name: sex
x_meaning: male
o_meaning: female
valid_x_min: 0.0
valid_x_max: 10.0
valid_y_min: 0.0
valid_y_max: 10.0
show_zero_bars: 0
```

### 2. Data Tables
Each dataset is stored in its own table (referenced by `metadata.table_name`).

**Schema Pattern:**
```sql
CREATE TABLE {table_name} (
    id INTEGER PRIMARY KEY,
    x REAL NOT NULL,
    y REAL NOT NULL,
    target TEXT NOT NULL
);
CREATE INDEX {table_name}_xy ON {table_name}(x, y);
CREATE INDEX {table_name}_target ON {table_name}(target);
```

**Field Descriptions:**
- `id`: Auto-incrementing primary key
- `x`: X coordinate (REAL)
- `y`: Y coordinate (REAL)
- `target`: Classification label (TEXT)

**Example Data:**
```
id  | x                | y                | target
----|------------------|------------------|--------
1   | 4.55696202531646 | 6.95652173913043 | male
2   | 4.43037974683544 | 6.52173913043478 | male
9   | 4.93670886075949 | 6.08695652173913 | female
10  | 4.93670886075949 | 6.95652173913043 | female
```

### 3. Unsaved Changes Table (Optional)
The `unsaved_changes` table tracks modifications for undo/redo functionality.

**Schema:**
```sql
CREATE TABLE unsaved_changes (
    id            INTEGER PRIMARY KEY,
    table_name    TEXT NOT NULL,
    action        TEXT NOT NULL CHECK (action IN ('insert','delete','update','meta')),
    data_id       INTEGER,
    x             REAL,
    y             REAL,
    old_target    TEXT,
    new_target    TEXT,
    meta_field    TEXT,
    old_value     TEXT,
    new_value     TEXT,
    is_active     INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX uc_table ON unsaved_changes(table_name, id);
```

**Note:** This table is used for edit history and may be empty or ignored when loading datasets.

## Usage for Narrative Learning

### Loading a DataPainter File
1. Open the SQLite database file
2. Query the `metadata` table to get available table names
3. For each table in metadata:
   - Read the table name and associated metadata
   - Query the data table to load points (id, x, y, target)

### Required Parameters
When using a DataPainter file with `standalone.py --dataset`:
- **Database file path**: Path to the .sqlite file
- **Table name**: Name of the table to use (from `metadata.table_name`)

### Data Format
- **Coordinates**: Two-dimensional (x, y) REAL values
- **Labels**: Text classification labels in the `target` column
- **Binary Classification**: Common pattern with two distinct target values
- **Multi-class**: Support for multiple target values possible

## Example Query Workflow

```sql
-- 1. Get available datasets
SELECT table_name, x_axis_name, y_axis_name, target_col_name
FROM metadata;

-- 2. Get metadata for specific dataset
SELECT * FROM metadata WHERE table_name = 'hamsters';

-- 3. Load all data points
SELECT id, x, y, target FROM hamsters;

-- 4. Count samples per class
SELECT target, COUNT(*) as count
FROM hamsters
GROUP BY target;
```

## Implementation Notes for standalone.py

1. **File Format Detection**: Check for `metadata` table existence to identify DataPainter files
2. **Table Selection**: If multiple tables exist, either:
   - Require `--table` parameter
   - Default to first table in metadata
   - Prompt user to select
3. **Data Conversion**: Map (x, y, target) to internal representation
4. **Metadata Usage**: Use axis names and meanings for display/explanation context
5. **Validation**: Check coordinate ranges against valid_x/y_min/max if needed
