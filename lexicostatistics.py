#!/usr/bin/env python3
"""Calculate lexical statistics for one or more language models combined."""
import argparse
import random

import datasetconfig
from modules.postgres import get_connection
from modules.text_analysis import calculate_zipfs_law, calculate_herdans_law


def compute_for_models(
    conn, training_models: list[str], sample_size: int, overwrite: bool
) -> None:
    """Compute statistics for one or more training models as an ensemble."""
    cur = conn.cursor()

    cur.execute(
        "SELECT model FROM models WHERE training_model = ANY(%s)",
        (training_models,),
    )
    model_rows = cur.fetchall()
    if not model_rows:
        raise SystemExit(
            "no models use training models " + ", ".join(training_models)
        )

    models = [row[0] for row in model_rows]

    cur.execute(
        """
        SELECT i.id, i.dataset, d.config_file
          FROM investigations i
          JOIN datasets d ON i.dataset = d.dataset
         WHERE i.model = ANY(%s)
        """,
        (models,),
    )
    investigations = cur.fetchall()
    if not investigations:
        raise SystemExit("no investigations found for those models")

    all_prompts = []
    all_reasoning = []
    for inv_id, dataset, config_file in investigations:
        cfg = datasetconfig.DatasetConfig(conn, config_file, dataset, inv_id)
        corpus = cfg.get_all_prompts_and_reasoning()
        prompts = corpus.get("prompts", [])
        reasoning = corpus.get("reasoning", [])
        for p, r in zip(prompts, reasoning):
            if p.strip() == "Choose randomly":
                continue
            if p:
                all_prompts.append(p)
            if r:
                all_reasoning.append(r)

    rng = random.Random(42)
    if len(all_prompts) > sample_size:
        all_prompts = rng.sample(all_prompts, sample_size)
    if len(all_reasoning) > sample_size:
        all_reasoning = rng.sample(all_reasoning, sample_size)

    prompt_zipf = calculate_zipfs_law(all_prompts)
    prompt_herdan = calculate_herdans_law(all_prompts)
    reasoning_zipf = calculate_zipfs_law(all_reasoning)
    reasoning_herdan = calculate_herdans_law(all_reasoning)

    for tm in training_models:
        if not overwrite:
            cur.execute(
                "SELECT 1 FROM lexicostatistics WHERE training_model = %s",
                (tm,),
            )
            if cur.fetchone():
                raise SystemExit(
                    f"results already exist for {tm}; use --overwrite"
                )

        cur.execute(
            """
            INSERT INTO lexicostatistics(
                training_model, prompt_zipf, prompt_zipf_r2,
                prompt_herdan, prompt_herdan_r2,
                reasoning_zipf, reasoning_zipf_r2,
                reasoning_herdan, reasoning_herdan_r2
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (training_model) DO UPDATE SET
                prompt_zipf = EXCLUDED.prompt_zipf,
                prompt_zipf_r2 = EXCLUDED.prompt_zipf_r2,
                prompt_herdan = EXCLUDED.prompt_herdan,
                prompt_herdan_r2 = EXCLUDED.prompt_herdan_r2,
                reasoning_zipf = EXCLUDED.reasoning_zipf,
                reasoning_zipf_r2 = EXCLUDED.reasoning_zipf_r2,
                reasoning_herdan = EXCLUDED.reasoning_herdan,
                reasoning_herdan_r2 = EXCLUDED.reasoning_herdan_r2,
                created = CURRENT_TIMESTAMP
            """,
            (
                tm,
                float(prompt_zipf["coefficient"]),
                float(prompt_zipf["r_squared"]),
                float(prompt_herdan["coefficient"]),
                float(prompt_herdan["r_squared"]),
                float(reasoning_zipf["coefficient"]),
                float(reasoning_zipf["r_squared"]),
                float(reasoning_herdan["coefficient"]),
                float(reasoning_herdan["r_squared"]),
            ),
        )

    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Herdan and Zipf coefficients for one or more training "
            "models"
        )
    )
    parser.add_argument(
        "training_models",
        help="Comma-separated language model names",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results",
    )
    parser.add_argument("--dsn", help="PostgreSQL DSN")
    parser.add_argument("--pg-config", help="JSON file containing postgres_dsn")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of prompt and reasoning texts to sample",
    )
    args = parser.parse_args()

    conn = get_connection(args.dsn, args.pg_config)

    training_models = [m.strip() for m in args.training_models.split(',') if m.strip()]
    compute_for_models(conn, training_models, args.sample_size, args.overwrite)

    conn.close()


if __name__ == "__main__":
    main()
