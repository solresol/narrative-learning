-- Add accuracy columns and round_completed timestamp for all *_rounds tables
--
-- This repository defines only six *_rounds tables:
--   espionage_rounds, potions_rounds, southgermancredit_rounds,
--   timetravel_insurance_rounds, titanic_rounds and wisconsin_rounds.
-- The loop below updates each of them if present, without affecting any other
-- tables.
DO $$
DECLARE
    r record;
BEGIN
    FOR r IN SELECT table_schema, table_name
             FROM information_schema.tables
             WHERE table_schema NOT IN ('pg_catalog','information_schema')
               AND table_name LIKE '%_rounds'
    LOOP
        EXECUTE format('ALTER TABLE %I.%I ADD COLUMN IF NOT EXISTS train_accuracy DOUBLE PRECISION;', r.table_schema, r.table_name);
        EXECUTE format('ALTER TABLE %I.%I ADD COLUMN IF NOT EXISTS validation_accuracy DOUBLE PRECISION;', r.table_schema, r.table_name);
        EXECUTE format('ALTER TABLE %I.%I ADD COLUMN IF NOT EXISTS test_accuracy DOUBLE PRECISION;', r.table_schema, r.table_name);
        EXECUTE format('ALTER TABLE %I.%I ADD COLUMN IF NOT EXISTS round_completed TIMESTAMP;', r.table_schema, r.table_name);
    END LOOP;
END $$;

