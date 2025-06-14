-- Schema for investigations
CREATE TABLE datasets (
    dataset TEXT PRIMARY KEY,
    config_file TEXT
);

CREATE TABLE models (
    model TEXT PRIMARY KEY,
    training_model TEXT,
    inference_model TEXT,
    example_count INTEGER DEFAULT 3,
    patience INTEGER DEFAULT 3
);

CREATE TABLE investigations (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset TEXT REFERENCES datasets(dataset),
    model TEXT REFERENCES models(model),
    sqlite_database TEXT,
    round_tracking_file TEXT,
    dump_file TEXT,
    round_number INTEGER DEFAULT 1
);
