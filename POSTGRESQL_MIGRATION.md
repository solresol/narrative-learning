# PostgreSQL Migration Notes

This project originally stored all results in individual SQLite files. We are
transitioning to PostgreSQL so multiple processes can share the same database
and to simplify backups. New scripts rely on a shared helper in
`modules/postgres.py` for creating a connection using either the `POSTGRES_DSN`
environment variable or a JSON configuration file specified via
`POSTGRES_CONFIG`.

Migration utilities:

- `update_round_number.py` reads the round tracking file referenced by an entry
  in the `investigations` table and updates the `round_number` column.
- `import_dataset.py` copies data from one or more legacy SQLite files into the
  PostgreSQL schema. Tables are created from an explicit SQL schema file (see
  `postgres-schemas/`) rather than being generated automatically. Dataset-
  specific tables (the core data and split assignments) are checked for
  consistency across all SQLite files. Only the `rounds` and `inferences`
  tables receive an extra `investigation_id` column when imported.

Run each script with either `--dsn` or `--config` to point to the PostgreSQL
instance.

