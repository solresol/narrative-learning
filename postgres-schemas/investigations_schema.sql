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

CREATE TABLE baseline_logreg (
    dataset TEXT REFERENCES datasets(dataset),
    feature TEXT,
    weight DOUBLE PRECISION,
    PRIMARY KEY (dataset, feature)
);

CREATE TABLE baseline_decision_tree (
    dataset TEXT PRIMARY KEY REFERENCES datasets(dataset),
    dot_data TEXT
);

CREATE TABLE baseline_dummy (
    dataset TEXT PRIMARY KEY REFERENCES datasets(dataset),
    constant_value TEXT
);

CREATE TABLE baseline_rulefit (
    dataset TEXT REFERENCES datasets(dataset),
    rule_index INTEGER,
    rule TEXT,
    weight DOUBLE PRECISION,
    PRIMARY KEY (dataset, rule_index)
);

CREATE TABLE baseline_bayesian_rule_list (
    dataset TEXT REFERENCES datasets(dataset),
    rule_order INTEGER,
    rule TEXT,
    probability DOUBLE PRECISION,
    PRIMARY KEY (dataset, rule_order)
);

CREATE TABLE baseline_corels (
    dataset TEXT REFERENCES datasets(dataset),
    rule_order INTEGER,
    rule TEXT,
    PRIMARY KEY (dataset, rule_order)
);

CREATE TABLE baseline_ebm (
    dataset TEXT REFERENCES datasets(dataset),
    feature TEXT,
    contribution_data JSONB,
    PRIMARY KEY (dataset, feature)
);
