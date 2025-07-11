-- Schema and data for model release dates

CREATE TABLE IF NOT EXISTS model_release_dates (
    training_model TEXT PRIMARY KEY,
    release_date DATE
);


-- Initial data
INSERT INTO model_release_dates(training_model, release_date) VALUES ('gemini-2.0-flash', '2024-12-12') ON CONFLICT (training_model) DO UPDATE SET release_date = EXCLUDED.release_date;
INSERT INTO model_release_dates(training_model, release_date) VALUES ('gpt-4o', '2024-05-13') ON CONFLICT (training_model) DO UPDATE SET release_date = EXCLUDED.release_date;
INSERT INTO model_release_dates(training_model, release_date) VALUES ('o3', '2025-04-16') ON CONFLICT (training_model) DO UPDATE SET release_date = EXCLUDED.release_date;
INSERT INTO model_release_dates(training_model, release_date) VALUES ('gpt-4.5-preview', '2025-02-27') ON CONFLICT (training_model) DO UPDATE SET release_date = EXCLUDED.release_date;
INSERT INTO model_release_dates(training_model, release_date) VALUES ('gpt-4o-mini', '2024-07-18') ON CONFLICT (training_model) DO UPDATE SET release_date = EXCLUDED.release_date;
INSERT INTO model_release_dates(training_model, release_date) VALUES ('gemini-2.5-pro-exp-03-25', '2025-03-25') ON CONFLICT (training_model) DO UPDATE SET release_date = EXCLUDED.release_date;
INSERT INTO model_release_dates(training_model, release_date) VALUES ('gemini-2.0-pro-exp', '2025-02-05') ON CONFLICT (training_model) DO UPDATE SET release_date = EXCLUDED.release_date;
INSERT INTO model_release_dates(training_model, release_date) VALUES ('claude-3-7-sonnet-20250219', '2025-02-19') ON CONFLICT (training_model) DO UPDATE SET release_date = EXCLUDED.release_date;
INSERT INTO model_release_dates(training_model, release_date) VALUES ('gpt-4.1', '2025-04-14') ON CONFLICT (training_model) DO UPDATE SET release_date = EXCLUDED.release_date;
INSERT INTO model_release_dates(training_model, release_date) VALUES ('claude-3-5-haiku-20241022', '2024-10-22') ON CONFLICT (training_model) DO UPDATE SET release_date = EXCLUDED.release_date;
INSERT INTO model_release_dates(training_model, release_date) VALUES ('o1', '2024-12-05') ON CONFLICT (training_model) DO UPDATE SET release_date = EXCLUDED.release_date;
INSERT INTO model_release_dates(training_model, release_date) VALUES ('gpt-3.5-turbo', '2023-03-01') ON CONFLICT (training_model) DO UPDATE SET release_date = EXCLUDED.release_date;
