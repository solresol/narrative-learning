CREATE TABLE patients (
  decodex INTEGER,
  Outcome text,
  Patient_ID text PRIMARY KEY,
  Histogen_Complex text,
  Sex text,
  Treatment_Months text,
  Genetic_Class_A_Matches text,
  Genetic_Class_B_Matches text,
  TcQ_mass text,
  Cohort text
);
CREATE TABLE inferences (
  round_id integer references rounds(round_id),
  creation_time datetime default current_timestamp,
  Patient_ID text references patients(Patient_ID),
  narrative_text text,
  llm_stderr text,
  prediction text,
  primary key (round_id, Patient_ID)
);
CREATE TABLE splits (
  split_id integer primary key autoincrement
);
CREATE TABLE sqlite_sequence(name,seq);
CREATE TABLE patient_splits (
  split_id integer references splits(split_id),
  Patient_ID text references patients(Patient_ID),
  holdout bool not null default false, -- holdout either for validation or test
  validation bool not null default false,
  primary key (split_id, Patient_ID)
);
CREATE TABLE rounds (
  round_id integer primary key autoincrement,
  round_start datetime default current_timestamp,
  split_id integer references splits(split_id),
  prompt text,
  reasoning_for_this_prompt text,
  stderr_from_prompt_creation text
);
