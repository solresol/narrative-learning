CREATE TABLE IF NOT EXISTS medical_treatment_data (
  PatientID TEXT primary key,
  Decodex INTEGER,
  Outcome TEXT,
  Group TEXT,
  Sex TEXT,
  Treatment Months REAL,
  Genetic Class A Matches INTEGER,
  Genetic Class B Matches INTEGER,
  TcQ mass REAL,
  Cohort TEXT
);

create table if not exists rounds (
  round_id integer primary key autoincrement,
  round_start datetime default current_timestamp,
  prompt text
);

create table if not exists inferences (
  round_id integer references rounds(round_id),
  creation_time datetime default current_timestamp,
  patientid text references medical_treatment_data(patientid),
  narrative_text text,
  prediction text
);
