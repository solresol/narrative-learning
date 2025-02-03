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

create table if not exists rounds (
  round_id integer primary key autoincrement,
  round_start datetime default current_timestamp,
  prompt text
);

create table if not exists inferences (
  round_id integer references rounds(round_id),
  creation_time datetime default current_timestamp,
  patient_id text references medical_treatment_data(patientid),
  narrative_text text,
  prediction text
);
