-- Remove investigations that rely on ollama-backed models and
-- reset round numbers for investigations with no inference data.
--
-- This script expects a temporary view ``empty_investigations(id)`` listing the
-- investigation IDs with no SQLite inference data. Generate it using::
--
--   python list_empty_investigations.py --skip-ollama --print-view > empty.sql
--   psql $DSN -f empty.sql -f cleanup_unrun_investigations.sql
--
-- Delete investigations using ollama-powered models that never produced data.
DELETE FROM investigations AS i
USING models AS m, empty_investigations AS e
WHERE i.id = e.id
  AND i.model = m.model
  AND m.training_model IN (
      'phi4:latest',
      'llama3.3:latest',
      'falcon3:1b',
      'falcon3:10b',
      'gemma2:27b',
      'gemma2:2b',
      'phi4-mini',
      'deepseek-r1:70b',
      'qwq:32b',
      'gemma3:27b',
      'cogito:70b'
  );

-- Clear round numbers for any investigation that has no data.
UPDATE investigations AS i
SET round_number = NULL
FROM empty_investigations AS e
WHERE i.id = e.id;
