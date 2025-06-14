CREATE TABLE coral_reefs (
  decodex INTEGER,
  CoralReefID text PRIMARY KEY,
  CurrentFlowQuality text,
  ObservationDuration text,
  ReefIntegrityScore text,
  PredatorActivityLevel text,
  AcousticIntensity text,
  AlgalCoverage text,
  CoralAgeEstimate text,
  BleachingEventsPerYear text,
  BiodiversityIndex text,
  NearbyHealthyReef text,
  ReefMonitoringDuration text,
  PollutionLevel text,
  ReefAverageAge text,
  DistantStressIndicators text,
  ReefDepthZone text,
  PreviousStressIncidents text,
  CoralDominantType text,
  SurveyorExperience text,
  RemoteSensorPresent text,
  InvasiveSpeciesDetected text,
  ReefHealthStatus text
);
CREATE TABLE inferences (
  round_id integer references rounds(round_id),
  creation_time datetime default current_timestamp,
  CoralReefID text references coral_reefs(CoralReefID),
  narrative_text text,
  llm_stderr text,
  prediction text,
  primary key (round_id, CoralReefID)
);
CREATE TABLE splits (
  split_id integer primary key autoincrement
);
CREATE TABLE sqlite_sequence(name,seq);
CREATE TABLE coral_reef_splits (
  split_id integer references splits(split_id),
  CoralReefID text references coral_reefs(CoralReefID),
  holdout bool not null default false, -- holdout either for validation or test
  validation bool not null default false,
  primary key (split_id, CoralReefID)
);
CREATE TABLE rounds (
  round_id integer primary key autoincrement,
  round_start datetime default current_timestamp,
  split_id integer references splits(split_id),
  prompt text,
  reasoning_for_this_prompt text,
  stderr_from_prompt_creation text
);
