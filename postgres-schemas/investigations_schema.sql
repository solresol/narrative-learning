-- Schema for investigations
CREATE TABLE datasets (
    dataset TEXT PRIMARY KEY,
    config_file TEXT
);

CREATE TABLE models (
    model TEXT PRIMARY KEY,
    training_model TEXT REFERENCES language_models(training_model),
    inference_model TEXT,
    example_count INTEGER DEFAULT 3,
    patience INTEGER DEFAULT 3
);

CREATE TABLE investigations (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset TEXT not null REFERENCES datasets(dataset),
    model TEXT not null REFERENCES models(model),
    sqlite_database TEXT,
    round_tracking_file TEXT,
    dump_file TEXT,
    round_number INTEGER,
    round_uuid UUID
);

CREATE TABLE baseline_results (
    dataset TEXT PRIMARY KEY REFERENCES datasets(dataset),
    logistic_regression DOUBLE PRECISION,
    decision_trees DOUBLE PRECISION,
    dummy DOUBLE PRECISION,
    rulefit DOUBLE PRECISION,
    bayesian_rule_list DOUBLE PRECISION,
    corels DOUBLE PRECISION,
    ebm DOUBLE PRECISION
);

CREATE TABLE dataset_provenance (
    dataset TEXT PRIMARY KEY REFERENCES datasets(dataset),
    provenance TEXT NOT NULL
);
