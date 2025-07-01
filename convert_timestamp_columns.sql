-- Convert existing timestamp columns to TIMESTAMPTZ assuming the original values are UTC.
-- The database timezone is set to Australia/Sydney so new timestamps use that zone.

ALTER DATABASE narrative SET timezone TO 'Australia/Sydney';

-- Espionage dataset
ALTER TABLE espionage_rounds
    ALTER COLUMN round_start TYPE TIMESTAMPTZ USING round_start AT TIME ZONE 'UTC',
    ALTER COLUMN round_start SET DEFAULT CURRENT_TIMESTAMP,
    ALTER COLUMN round_completed TYPE TIMESTAMPTZ USING round_completed AT TIME ZONE 'UTC';
ALTER TABLE espionage_inferences
    ALTER COLUMN creation_time TYPE TIMESTAMPTZ USING creation_time AT TIME ZONE 'UTC',
    ALTER COLUMN creation_time SET DEFAULT CURRENT_TIMESTAMP;

-- Potions dataset
ALTER TABLE potions_rounds
    ALTER COLUMN round_start TYPE TIMESTAMPTZ USING round_start AT TIME ZONE 'UTC',
    ALTER COLUMN round_start SET DEFAULT CURRENT_TIMESTAMP,
    ALTER COLUMN round_completed TYPE TIMESTAMPTZ USING round_completed AT TIME ZONE 'UTC';
ALTER TABLE potions_inferences
    ALTER COLUMN creation_time TYPE TIMESTAMPTZ USING creation_time AT TIME ZONE 'UTC',
    ALTER COLUMN creation_time SET DEFAULT CURRENT_TIMESTAMP;

-- Southgermancredit dataset
ALTER TABLE southgermancredit_rounds
    ALTER COLUMN round_start TYPE TIMESTAMPTZ USING round_start AT TIME ZONE 'UTC',
    ALTER COLUMN round_start SET DEFAULT CURRENT_TIMESTAMP,
    ALTER COLUMN round_completed TYPE TIMESTAMPTZ USING round_completed AT TIME ZONE 'UTC';
ALTER TABLE southgermancredit_inferences
    ALTER COLUMN creation_time TYPE TIMESTAMPTZ USING creation_time AT TIME ZONE 'UTC',
    ALTER COLUMN creation_time SET DEFAULT CURRENT_TIMESTAMP;

-- Timetravel insurance dataset
ALTER TABLE timetravel_insurance_rounds
    ALTER COLUMN round_start TYPE TIMESTAMPTZ USING round_start AT TIME ZONE 'UTC',
    ALTER COLUMN round_start SET DEFAULT CURRENT_TIMESTAMP,
    ALTER COLUMN round_completed TYPE TIMESTAMPTZ USING round_completed AT TIME ZONE 'UTC';
ALTER TABLE timetravel_insurance_inferences
    ALTER COLUMN creation_time TYPE TIMESTAMPTZ USING creation_time AT TIME ZONE 'UTC',
    ALTER COLUMN creation_time SET DEFAULT CURRENT_TIMESTAMP;

-- Titanic dataset
ALTER TABLE titanic_rounds
    ALTER COLUMN round_start TYPE TIMESTAMPTZ USING round_start AT TIME ZONE 'UTC',
    ALTER COLUMN round_start SET DEFAULT CURRENT_TIMESTAMP,
    ALTER COLUMN round_completed TYPE TIMESTAMPTZ USING round_completed AT TIME ZONE 'UTC';
ALTER TABLE titanic_inferences
    ALTER COLUMN creation_time TYPE TIMESTAMPTZ USING creation_time AT TIME ZONE 'UTC',
    ALTER COLUMN creation_time SET DEFAULT CURRENT_TIMESTAMP;

-- Wisconsin dataset
ALTER TABLE wisconsin_rounds
    ALTER COLUMN round_start TYPE TIMESTAMPTZ USING round_start AT TIME ZONE 'UTC',
    ALTER COLUMN round_start SET DEFAULT CURRENT_TIMESTAMP,
    ALTER COLUMN round_completed TYPE TIMESTAMPTZ USING round_completed AT TIME ZONE 'UTC';
ALTER TABLE wisconsin_inferences
    ALTER COLUMN creation_time TYPE TIMESTAMPTZ USING creation_time AT TIME ZONE 'UTC',
    ALTER COLUMN creation_time SET DEFAULT CURRENT_TIMESTAMP;
