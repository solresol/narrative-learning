import json
import math
from typing import List, Dict
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from datasetconfig import DatasetConfig
from modules.text_analysis import count_words


def load_results_dataframe(conn, dataset: str, cfg_file: str, model_details_file: str = "model_details.json") -> pd.DataFrame:
    """Return a DataFrame similar to the CSV created by create_task_csv_file."""
    with open(model_details_file, "r", encoding="utf-8") as f:
        model_details = json.load(f)

    cur = conn.cursor()
    cur.execute(
        """
        SELECT i.id, i.model, m.training_model, m.patience, m.example_count
          FROM investigations i
          JOIN models m ON i.model = m.model
         WHERE i.dataset = %s
         ORDER BY i.id
        """,
        (dataset,),
    )
    investigations = cur.fetchall()

    cfg = DatasetConfig(conn, cfg_file, dataset)
    dataset_size = cfg.get_data_point_count()

    cur.execute(
        """
        SELECT logistic_regression, decision_trees, dummy, rulefit,
               bayesian_rule_list, corels, ebm
          FROM baseline_results
         WHERE dataset = %s
        """,
        (dataset,),
    )
    baseline_row = cur.fetchone()
    baseline_cols = [
        "logistic regression",
        "decision trees",
        "dummy",
        "rulefit",
        "bayesian rule list",
        "corels",
        "ebm",
    ]
    baseline = dict(zip(baseline_cols, baseline_row)) if baseline_row else {}

    rows: List[Dict] = []
    for inv_id, run_name, training_model, patience, sampler in investigations:
        cfg = DatasetConfig(conn, cfg_file, dataset, inv_id)
        split_id = cfg.get_latest_split_id()
        try:
            best_round = cfg.get_best_round_id(split_id, "accuracy")
            val_df = cfg.generate_metrics_data(split_id, "accuracy", "validation")
            val_acc = val_df[val_df.round_id == best_round].metric.iloc[0]
            test_acc = cfg.get_test_metric_for_best_validation_round(split_id, "accuracy")
        except Exception:
            val_acc = test_acc = None
            best_round = None
        prompt_word_count = reasoning_word_count = cumul_reasoning_words = 0
        herdan_coeff = herdan_r2 = zipf_coeff = zipf_r2 = None
        if best_round is not None:
            try:
                prompt = cfg.get_round_prompt(best_round)
                reasoning = cfg.get_round_reasoning(best_round)
                prompt_word_count = count_words(prompt)
                reasoning_word_count = count_words(reasoning)
                totals = cfg.get_total_word_count(split_id, best_round)
                cumul_reasoning_words = totals["reasoning_words"]
                herdan = cfg.calculate_herdans_law(split_id)
                zipf = cfg.calculate_zipfs_law(split_id)
                herdan_coeff = herdan["coefficient"]
                herdan_r2 = herdan["r_squared"]
                zipf_coeff = zipf["coefficient"]
                zipf_r2 = zipf["r_squared"]
            except Exception:
                pass
        model_size = model_details.get(run_name, {}).get("parameters", "")
        lower_bound = None
        neg_log_error = None
        if test_acc is not None:
            count_correct = round(test_acc * dataset_size)
            lower_bound, _ = proportion_confint(count_correct, dataset_size, alpha=0.05, method="beta")
            neg_log_error = -math.log10(1 - lower_bound) if lower_bound < 1 else float("inf")
        rows.append({
            "Task": dataset,
            "Model": training_model,
            "Run Name": f"{run_name}.",
            "Patience": patience,
            "Sampler": sampler,
            "Accuracy": test_acc,
            "Accuracy Lower Bound": lower_bound,
            "Neg Log Error": neg_log_error,
            "Rounds": best_round,
            "Prompt Word Count": prompt_word_count,
            "Reasoning Word Count": reasoning_word_count,
            "Cumulative Reasoning Words": cumul_reasoning_words,
            "Herdan Coefficient": herdan_coeff,
            "Herdan R-squared": herdan_r2,
            "Zipf Coefficient": zipf_coeff,
            "Zipf R-squared": zipf_r2,
            "Model Size": model_size,
            "Data Points": dataset_size,
            **baseline,
        })

    return pd.DataFrame(rows)
