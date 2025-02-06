CREATE TABLE IF NOT EXISTS medical_treatment_data (
  Patient_ID TEXT primary key,
  Decodex INTEGER,
  Outcome TEXT,
  Treatment_Group TEXT,
  Sex TEXT,
  Treatment_Months REAL,
  Genetic_Class_A_Matches INTEGER,
  Genetic_Class_B_Matches INTEGER,
  TcQ_mass REAL,
  Cohort TEXT
);


create table if not exists inferences (
  round_id integer references rounds(round_id),
  creation_time datetime default current_timestamp,
  patient_id text references medical_treatment_data(patientid),
  narrative_text text,
  llm_stderr text,
  prediction text,
  primary key (round_id, patient_id)
);

create table if not exists splits (
  split_id integer primary key autoincrement
);

create table if not exists patient_split (
  split_id integer references splits(split_id),
  patient_id text references medical_treatment_data(patient_id),
  holdout bool not null default false,
  primary key (split_id, patient_id)
);

create table if not exists rounds (
  round_id integer primary key autoincrement,
  round_start datetime default current_timestamp,
  split_id integer references splits(split_id),
  prompt text,
  reasoning_for_this_prompt text,
  stderr_from_prompt_creation text
);

