CREATE TABLE espionage_agents (
        AgentID TEXT PRIMARY KEY,
        SecretHandshakeQuality REAL,
        AccentThickness REAL,
        AgentStatus TEXT
    );
CREATE TABLE espionage_splits (
        split_id INTEGER,
        AgentID TEXT REFERENCES espionage_agents(AgentID),
        holdout BOOLEAN NOT NULL DEFAULT FALSE,
        validation BOOLEAN NOT NULL DEFAULT FALSE,
        PRIMARY KEY (split_id, AgentID)
    );
CREATE TABLE splits (
        split_id INTEGER PRIMARY KEY,
        name TEXT
    );
CREATE TABLE inferences (
          round_id integer references rounds(round_id),
          creation_time datetime default current_timestamp,
          AgentID text references espionage_agents(AgentID),
          narrative_text text,
          llm_stderr text,
          prediction text,
          primary key (round_id, AgentID)
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
