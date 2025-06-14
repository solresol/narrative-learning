CREATE TABLE time_travel_incidents (
        IncidentID TEXT PRIMARY KEY,
        TimelineDeviation REAL,
        ParadoxCount REAL,
        PolicyClaim TEXT
    );
CREATE TABLE time_travel_splits (
        split_id INTEGER,
        IncidentID TEXT REFERENCES time_travel_incidents(IncidentID),
        holdout BOOLEAN NOT NULL DEFAULT FALSE,
        validation BOOLEAN NOT NULL DEFAULT FALSE,
        PRIMARY KEY (split_id, IncidentID)
    );
CREATE TABLE splits (
        split_id INTEGER PRIMARY KEY,
        name TEXT
    );
CREATE TABLE inferences (
          round_id integer references rounds(round_id),
          creation_time datetime default current_timestamp,
          IncidentID text references time_travel_incidents(IncidentID),
          narrative_text text,
          llm_stderr text,
          prediction text,
          primary key (round_id, IncidentID)
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
