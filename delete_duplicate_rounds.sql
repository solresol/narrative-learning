-- Delete duplicate rounds for each investigation across all *_rounds tables.
-- For investigations with multiple rounds, keep only the earliest round_id.
DO $$
DECLARE
    r record;
BEGIN
    FOR r IN
        SELECT table_schema, table_name
          FROM information_schema.tables
         WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
           AND table_name LIKE '%_rounds'
    LOOP
        EXECUTE format(
            $$DELETE FROM %I.%I t
              USING (
                  SELECT investigation_id, MIN(round_id) AS keep_id
                    FROM %I.%I
                   GROUP BY investigation_id
                   HAVING COUNT(*) > 1
              ) k
             WHERE t.investigation_id = k.investigation_id
               AND t.round_id <> k.keep_id;$$,
            r.table_schema, r.table_name,
            r.table_schema, r.table_name
        );
    END LOOP;
END $$;
