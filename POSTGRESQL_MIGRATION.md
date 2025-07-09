# PostgreSQL Migration Notes

This project originally stored all results in individual SQLite files. We are
transitioning to PostgreSQL so multiple processes can share the same database
and to simplify backups. New scripts rely on a shared helper in
`modules/postgres.py` for creating a connection. It uses the standard
PG environment variables to manage the connection.

Migration utilities:

- `update_round_number.py` reads the round tracking file referenced by an entry
  in the `investigations` table and updates the `round_number` column.
- `import_dataset.py` copies data from legacy SQLite files into the
  PostgreSQL schema. The script now only needs an investigation ID and will
  look up the dataset name, configuration file and default SQLite path from the
  `investigations` and `datasets` tables. If no SQLite path is given it uses the
  one recorded in the table and loads the appropriate schema from
  `postgres-schemas/`. Dataset-specific tables (the core data and split
  assignments) are checked for consistency across all SQLite files. Only the
  `rounds` and `inferences` tables receive an extra `investigation_id` column
  when imported. Because each SQLite database numbers rounds starting from 1,
  the import script lets PostgreSQL assign new ``round_id`` values and updates
  the corresponding ``inferences`` rows so that duplicates do not cause
  conflicts.


