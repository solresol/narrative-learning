CREATE TABLE magic_potions (
        PotionBatchID TEXT PRIMARY KEY,
        FizzIntensity REAL,
        ColourShift REAL,
        PotionEffectiveness TEXT
    );
CREATE TABLE potion_splits (
        split_id INTEGER,
        PotionBatchID TEXT REFERENCES magic_potions(PotionBatchID),
        holdout BOOLEAN NOT NULL DEFAULT FALSE,
        validation BOOLEAN NOT NULL DEFAULT FALSE,
        PRIMARY KEY (split_id, PotionBatchID)
    );
CREATE TABLE splits (
        split_id INTEGER PRIMARY KEY,
        name TEXT
    );
CREATE TABLE inferences (
          round_id integer references rounds(round_id),
          creation_time datetime default current_timestamp,
          PotionBatchID text references magic_potions(PotionBatchID),
          narrative_text text,
          llm_stderr text,
          prediction text,
          primary key (round_id, PotionBatchID)
        );
CREATE TABLE rounds (
         round_id integer primary key autoincrement,
         round_start datetime default current_timestamp,
         split_id integer references splits(split_id),
         prompt text,
         reasoning_for_this_prompt text,
         stderr_from_prompt_creation text
      );
CREATE TABLE sqlite_sequence(name,seq);
