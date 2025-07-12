-- Schema for aggregated lexical statistics per language model
CREATE TABLE IF NOT EXISTS lexicostatistics (
    training_model TEXT PRIMARY KEY,
    prompt_zipf DOUBLE PRECISION,
    prompt_zipf_r2 DOUBLE PRECISION,
    prompt_herdan DOUBLE PRECISION,
    prompt_herdan_r2 DOUBLE PRECISION,
    reasoning_zipf DOUBLE PRECISION,
    reasoning_zipf_r2 DOUBLE PRECISION,
    reasoning_herdan DOUBLE PRECISION,
    reasoning_herdan_r2 DOUBLE PRECISION,
    created TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
