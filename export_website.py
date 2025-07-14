#!/usr/bin/env python3
"""Generate a static HTML representation of the investigations database."""
from __future__ import annotations
import os
import html
import argparse
from datetime import datetime
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress
import numpy as np

import pandas as pd

from modules.postgres import get_connection, get_investigation_settings
from modules.results_loader import load_results_dataframe
from datasetconfig import DatasetConfig
from chartutils import draw_baselines
from modules.ensemble_selection import get_interesting_ensembles
from modules.metrics import accuracy_to_kt
import math


def write_page(path: str, title: str, body: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'>")
        f.write(f"<title>{html.escape(title)}</title>")
        f.write("</head><body>")
        f.write(f"<h1>{html.escape(title)}</h1>\n")
        f.write(body)
        f.write("</body></html>")


def get_split_id(cfg: DatasetConfig) -> int:
    cur = cfg.conn.cursor()
    cfg._execute(cur, f"SELECT MIN(split_id) FROM {cfg.splits_table}")
    row = cur.fetchone()
    return row[0]


def plot_release_chart(
    conn, dataset: str, df: pd.DataFrame, out_path: str, dataset_size: int
) -> Optional[tuple[float, float, float]]:
    """Plot test scores by model release date for a dataset.

    Returns the slope, intercept and p-value of the regression line calculated
    over the ensemble maximum scores, or ``None`` if no ensemble data exists.
    """
    if df.empty:
        return None
    cur = conn.cursor()
    cur.execute(
        "SELECT training_model, release_date, ollama_hosted FROM language_models"
    )
    info = pd.DataFrame(cur.fetchall(), columns=["Model", "Release Date", "ollama"])
    df = df.merge(info, on="Model", how="left")
    # ``ollama`` is True for models hosted on Ollama. These rows should be
    # excluded from the plot, treating missing values as False.
    df = df[~df["ollama"].fillna(False).astype(bool)]
    df.dropna(subset=["Release Date", "Accuracy"], inplace=True)
    if df.empty:
        raise RuntimeError("no model results found to plot")

    df["Release Date"] = pd.to_datetime(df["Release Date"])
    df.sort_values("Release Date", inplace=True)
    df["KT"] = df["Accuracy"].apply(lambda x: accuracy_to_kt(x, dataset_size))

    cur.execute(
        """
        SELECT logistic_regression, decision_trees, dummy, rulefit,
               bayesian_rule_list, corels, ebm
          FROM baseline_results
         WHERE dataset = %s
        """,
        (dataset,),
    )
    row = cur.fetchone()
    if row:
        cols_db = [
            "logistic_regression",
            "decision_trees",
            "dummy",
            "rulefit",
            "bayesian_rule_list",
            "corels",
            "ebm",
        ]
        cols_df = [
            "logistic regression",
            "decision trees",
            "dummy",
            "rulefit",
            "bayesian rule list",
            "corels",
            "ebm",
        ]
        for col_db, col_df, val in zip(cols_db, cols_df, row):
            df[col_df] = val

    ens_df = get_interesting_ensembles(conn, dataset)
    if ens_df.empty:
        raise RuntimeError("no interesting ensembles found")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df["Release Date"], -df["KT"], marker="o", label="model")
    ax.set_xlabel("Model release date")
    ax.set_ylabel("log10 KT accuracy")
    draw_baselines(ax, df, xpos=df["Release Date"].max(), dataset_size=dataset_size)

    ens_df = ens_df.copy()
    ens_df["Release Date"] = pd.to_datetime(ens_df["release_date"])
    ens_df.sort_values("Release Date", inplace=True)
    ens_df["KT"] = (
        ens_df["test_correct"] / ens_df["test_total"]
    ).apply(lambda x: accuracy_to_kt(x, dataset_size))
    ax.scatter(
        ens_df["Release Date"],
        -ens_df["KT"],
        marker="x",
        c="red",
        label="ensemble",
    )
    x = mdates.date2num(ens_df["Release Date"])
    y = -ens_df["KT"]
    if len(ens_df) > 1:
        slope, intercept, r, pval, std = linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(mdates.num2date(xs), intercept + slope * xs, "--", c="red")
    else:
        slope = intercept = pval = float("nan")

    ax.set_title("Test scores by release date")

    ax.legend()
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    return slope, intercept, pval


def generate_round_page(cfg: DatasetConfig, investigation_id: int, round_id: int, out_dir: str) -> None:
    cur = cfg.conn.cursor()
    rounds_table = cfg.rounds_table
    inf_table = f"{cfg.dataset}_inferences" if cfg.dataset else "inferences"
    cfg._execute(
        cur,
        f"""
        SELECT round_uuid, prompt, train_accuracy, validation_accuracy,
               test_accuracy, round_completed, round_start
          FROM {rounds_table}
         WHERE round_id = ? AND investigation_id = ?
        """,
        (round_id, investigation_id),
    )
    row = cur.fetchone()
    if not row:
        return
    (uuid, prompt, train_acc, val_acc, test_acc, completed, created) = row
    cfg._execute(
        cur,
        f"SELECT COUNT(*), MIN(creation_time), MAX(creation_time) FROM {inf_table} WHERE round_id = ? AND investigation_id = ?",
        (round_id, investigation_id),
    )
    inf_count, first_inf, last_inf = cur.fetchone()
    body = []
    body.append(f"<p><b>Round UUID:</b> {uuid}</p>")
    body.append(f"<p><b>Prompt:</b><br><pre>{html.escape(prompt)}</pre></p>")
    body.append("<ul>")
    body.append(f"<li>Inferences: {inf_count}</li>")
    body.append(f"<li>Created: {created}</li>")
    body.append(f"<li>First inference: {first_inf if first_inf else 'n/a'}</li>")
    body.append(f"<li>Last inference: {last_inf if last_inf else 'n/a'}</li>")
    body.append(f"<li>Completed: {'yes' if completed else 'no'}</li>")
    body.append(f"<li>Train accuracy: {train_acc if train_acc is not None else 'n/a'}</li>")
    body.append(f"<li>Validation accuracy: {val_acc if val_acc is not None else 'n/a'}</li>")
    body.append(f"<li>Test accuracy: {test_acc if test_acc is not None else 'n/a'}</li>")
    body.append("</ul>")
    write_page(os.path.join(out_dir, "index.html"), f"Round {round_id}", "\n".join(body))


def generate_investigation_page(
    conn,
    dataset: str,
    cfg_file: str,
    inv_id: int,
    base_dir: str,
    progress: bool = False,
) -> None:
    inv_dir = os.path.join(base_dir, "investigation", str(inv_id))
    os.makedirs(inv_dir, exist_ok=True)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT i.round_number, m.model, m.training_model, m.inference_model,
               m.example_count, m.patience
          FROM investigations i
          JOIN models m ON i.model = m.model
         WHERE i.id = %s
        """,
        (inv_id,),
    )
    row = cur.fetchone()
    if not row:
        return
    round_no, model, training_model, inference_model, example_count, patience = row

    cfg = DatasetConfig(conn, cfg_file, dataset, inv_id)
    split_id = cfg.get_latest_split_id()
    try:
        best_round = cfg.get_best_round_id(split_id, "accuracy")
    except Exception:
        best_round = None
    inf_table = f"{cfg.dataset}_inferences" if cfg.dataset else "inferences"
    cur.execute(
        f"""
        SELECT r.round_id, r.round_uuid, r.round_start, r.round_completed,
               r.train_accuracy, r.validation_accuracy, r.test_accuracy,
               COUNT(i.*) AS inf_count
          FROM {cfg.rounds_table} r
          LEFT JOIN {inf_table} i ON (
                r.round_id = i.round_id AND
                r.investigation_id = i.investigation_id)
         WHERE r.investigation_id = %s
         GROUP BY r.round_id, r.round_uuid, r.round_start, r.round_completed,
                  r.train_accuracy, r.validation_accuracy, r.test_accuracy
         ORDER BY r.round_start
        """,
        (inv_id,),
    )
    rounds = cur.fetchall()
    if rounds and rounds[-1][-1] == 0:
        rounds = rounds[:-1]

    df_rounds = pd.DataFrame(
        rounds,
        columns=[
            "round_id",
            "round_uuid",
            "round_start",
            "round_completed",
            "train_acc",
            "val_acc",
            "test_acc",
            "inf_count",
        ],
    )
    body = ["<ul>"]
    body.append(f"<li>Model: {model}</li>")
    body.append(f"<li>Training model: {training_model}</li>")
    body.append(f"<li>Inference model: {inference_model}</li>")
    body.append(f"<li>Example count: {example_count}</li>")
    body.append(f"<li>Patience: {patience}</li>")
    body.append(f"<li>Current round: {round_no}</li>")
    if best_round is not None:
        body.append(f"<li>Best round: {best_round}</li>")
    body.append("</ul>")

    body.append("<h2>Rounds</h2>")
    dataset_size = cfg.get_data_point_count()
    body.append("<table border='1'>")
    body.append(
        "<tr><th>Round</th><th>UUID</th><th>Started</th><th>Completed" +
        "</th><th>Val Acc</th><th>Val KT</th><th>Test Acc</th><th>Test KT</th></tr>"
    )
    iterator = rounds
    if progress:
        import tqdm
        iterator = tqdm.tqdm(rounds, desc=f"investigation {inv_id}")
    for r_id, r_uuid, r_start, r_completed, train_acc, v_acc, t_acc, inf_count in iterator:
        link = f"round/{r_id}/index.html"
        highlight = " style='background-color:#ffffcc'" if best_round == r_id else ""
        if v_acc is not None:
            v_acc_disp = f"{v_acc:.3f}"
            v_kt = f"{accuracy_to_kt(v_acc, dataset_size):.3f}"
        else:
            v_acc_disp = v_kt = "n/a"
        if t_acc is not None:
            t_acc_disp = f"{t_acc:.3f}"
            t_kt = f"{accuracy_to_kt(t_acc, dataset_size):.3f}"
        else:
            t_acc_disp = t_kt = "n/a"
        body.append(
            f"<tr{highlight}><td><a href='{link}'>{r_id}</a></td><td>{r_uuid}</td><td>{r_start:%Y-%m-%d}</td><td>{r_completed if r_completed else 'in progress'}</td><td>{v_acc_disp}</td><td>{v_kt}</td><td>{t_acc_disp}</td><td>{t_kt}</td></tr>"
        )
        generate_round_page(cfg, inv_id, r_id, os.path.join(inv_dir, "round", str(r_id)))
    body.append("</table>")

    if df_rounds.empty:
        raise RuntimeError("no rounds data found")
    plot_df = (
        df_rounds.dropna(subset=["val_acc"]).sort_values("round_start").reset_index(drop=True)
    )
    plot_df["rank"] = plot_df.index + 1
    for col in ["train_acc", "val_acc", "test_acc"]:
        plot_df[col] = plot_df[col].apply(lambda x: -accuracy_to_kt(x, dataset_size))
    plt.figure(figsize=(8,4))
    plt.plot(plot_df["rank"], plot_df["train_acc"], label="train")
    plt.plot(plot_df["rank"], plot_df["val_acc"], label="validation")
    plt.plot(plot_df["rank"], plot_df["test_acc"], label="test")
    plt.xticks(plot_df["rank"], plot_df["round_id"])
    plt.xlabel("Round")
    plt.ylabel("log10 KT accuracy")
    plt.title("Round Scores")
    plt.legend()
    plt.tight_layout()
    chart_path = os.path.join(inv_dir, "scores.png")
    plt.savefig(chart_path)
    plt.close()
    body.append("<h2>Scores</h2>")
    body.append(f"<img src='scores.png' alt='round scores'>")

    write_page(os.path.join(inv_dir, "index.html"), f"Investigation {inv_id}", "\n".join(body))


def generate_dataset_page(
    conn, dataset: str, cfg_file: str, out_dir: str, progress: bool = False
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT i.id, i.model, m.training_model
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

    body = []
    cur.execute(
        "SELECT provenance FROM dataset_provenance WHERE dataset = %s",
        (dataset,),
    )
    row = cur.fetchone()
    if row:
        body.append("<h2>Provenance</h2>")
        body.append(f"<pre>{html.escape(row[0])}</pre>")

    body.append("<h2>Investigations</h2>")
    groups: dict[str, list[tuple[int, str]]] = {}
    iterator = investigations
    if progress:
        import tqdm
        iterator = tqdm.tqdm(investigations, desc=f"dataset {dataset}")
    for inv_id, model, training_model in iterator:
        groups.setdefault(training_model, []).append((inv_id, model))
        generate_investigation_page(conn, dataset, cfg_file, inv_id, out_dir, progress)

    for tm, rows in groups.items():
        body.append(f"<h3>{tm}</h3><ul>")
        for inv_id, model in rows:
            body.append(f"<li><a href='investigation/{inv_id}/index.html'>{inv_id} ({model})</a></li>")
        body.append("</ul>")

    # Load results directly from the database
    df_results = load_results_dataframe(conn, dataset, cfg_file)

    # Plot test results by model release date
    chart_file = os.path.join(out_dir, "release_scores.png")
    stats = plot_release_chart(conn, dataset, df_results, chart_file, dataset_size)
    if os.path.exists(chart_file):
        body.append("<h2>Test scores by release date</h2>")
        body.append(f"<img src='release_scores.png' alt='scores by release date'>")
        if stats is not None:
            slope, intercept, pval = stats
            body.append(
                f"<p>Regression slope: {slope:.4f}, intercept: {intercept:.4f}, p-value: {pval:.3g}</p>"
            )

    if not df_results.empty:
        df = df_results.copy()
        cur.execute(
            "SELECT training_model, release_date, ollama_hosted FROM language_models"
        )
        info = pd.DataFrame(cur.fetchall(), columns=["Model", "Release Date", "ollama"])
        df = df.merge(info, on="Model", how="left")
        df = df[~df["ollama"].fillna(False).astype(bool)]
        df.dropna(subset=["Release Date", "Accuracy"], inplace=True)
        df["Release Date"] = pd.to_datetime(df["Release Date"])
        df.sort_values("Release Date", inplace=True)
        df["KT"] = df["Accuracy"].apply(lambda x: accuracy_to_kt(x, dataset_size))
        body.append("<h2>Model Scores</h2><table border='1'>")
        body.append(
            "<tr><th>Model</th><th>Release Date</th><th>Examples</th><th>Patience" +
            "</th><th>Val Acc</th><th>Val KT</th><th>Test Acc</th><th>Test KT</th></tr>"
        )
        for m, d_, run_name, ex, patience in df[["Model", "Release Date", "Run Name", "Sampler", "Patience"]].itertuples(index=False):
            cur.execute(
                "SELECT id FROM investigations WHERE dataset = %s AND model = %s",
                (dataset, run_name.rstrip(".")),
            )
            inv_row = cur.fetchone()
            if inv_row:
                cfg = DatasetConfig(conn, cfg_file, dataset, inv_row[0])
                split_id = cfg.get_latest_split_id()
                try:
                    best_round = cfg.get_best_round_id(split_id, "accuracy")
                    val_df = cfg.generate_metrics_data(split_id, "accuracy", "validation")
                    val_acc = val_df[val_df.round_id == best_round].metric.iloc[0]
                    test_acc = cfg.get_test_metric_for_best_validation_round(split_id, "accuracy")
                except Exception:
                    val_acc = test_acc = None
            else:
                val_acc = test_acc = None
            val_acc_disp = f"{val_acc:.3f}" if val_acc is not None else "n/a"
            val_kt = f"{accuracy_to_kt(val_acc, dataset_size):.3f}" if val_acc is not None else "n/a"
            test_acc_disp = f"{test_acc:.3f}" if test_acc is not None else "n/a"
            test_kt = f"{accuracy_to_kt(test_acc, dataset_size):.3f}" if test_acc is not None else "n/a"
            body.append(
                f"<tr><td>{m}</td><td>{d_.date()}</td><td>{ex}</td><td>{patience}</td>"
                f"<td>{val_acc_disp}</td><td>{val_kt}</td><td>{test_acc_disp}</td><td>{test_kt}</td></tr>"
            )
        body.append("</table>")

        ens_df = get_interesting_ensembles(conn, dataset)
        if ens_df.empty:
            raise RuntimeError("no ensemble data found")
        ens_df = ens_df.copy()
        ens_df["Release Date"] = pd.to_datetime(ens_df["release_date"])
        ens_df.sort_values("Release Date", inplace=True)
        ens_df["validation_kt"] = ens_df["validation_accuracy"].apply(lambda x: accuracy_to_kt(x, dataset_size))
        ens_df["test_accuracy"] = ens_df["test_correct"] / ens_df["test_total"]
        ens_df["test_kt"] = ens_df["test_accuracy"].apply(lambda x: accuracy_to_kt(x, dataset_size))
        body.append("<h2>Ensemble Max Scores</h2><table border='1'>")
        body.append(
            "<tr><th>Release Date</th><th>Val Acc</th><th>Val KT</th><th>Test Acc</th><th>Test KT</th><th>Ensemble</th></tr>"
        )
        for d_, v_acc, v_kt, t_acc, t_kt, names in ens_df[[
            "Release Date",
            "validation_accuracy",
            "validation_kt",
            "test_accuracy",
            "test_kt",
            "model_names",
        ]].itertuples(index=False):
            body.append(
                f"<tr><td>{d_.date()}</td><td>{v_acc:.3f}</td><td>{v_kt:.3f}</td><td>{t_acc:.3f}</td><td>{t_kt:.3f}</td><td>{html.escape(names)}</td></tr>"
            )
        body.append("</table>")

    cur.execute(
        """
        SELECT logistic_regression, decision_trees, dummy, rulefit,
               bayesian_rule_list, corels, ebm
          FROM baseline_results
         WHERE dataset = %s
        """,
        (dataset,),
    )
    row = cur.fetchone()
    if row:
        names = [
            "Logistic regression",
            "Decision trees",
            "Dummy",
            "RuleFit",
            "BayesianRuleList",
            "CORELS",
            "EBM",
        ]
        body.append("<h2>Baseline</h2><table border='1'>")
        body.append("<tr><th>Model</th><th>Accuracy</th><th>KT</th></tr>")
        for name, val in zip(names, row):
            if val is not None:
                kt = accuracy_to_kt(val, dataset_size)
                display_kt = f"{kt:.3f}"
                display_acc = f"{val:.3f}"
            else:
                display_kt = display_acc = "n/a"
            body.append(f"<tr><td>{name}</td><td>{display_acc}</td><td>{display_kt}</td></tr>")
        body.append("</table>")

    write_page(os.path.join(out_dir, "index.html"), f"Dataset {dataset}", "\n".join(body))


def generate_model_page(conn, model: str, out_dir: str, dataset_lookup: dict[str, str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cur = conn.cursor()
    cur.execute("SELECT training_model, inference_model FROM models WHERE model = %s", (model,))
    row = cur.fetchone()
    if not row:
        return
    training_model, inference_model = row
    cur.execute("SELECT id, dataset FROM investigations WHERE model = %s ORDER BY id", (model,))
    investigations = cur.fetchall()
    perf_rows: List[Tuple[str, str, str]] = []
    for inv_id, dataset in investigations:
        cfg = DatasetConfig(conn, dataset_lookup[dataset], dataset, inv_id)
        split_id = cfg.get_latest_split_id()
        try:
            best_round = cfg.get_best_round_id(split_id, "accuracy")
            val_df = cfg.generate_metrics_data(split_id, "accuracy", "validation")
            val_acc = val_df[val_df.round_id == best_round].metric.iloc[0]
            try:
                test_acc = cfg.get_test_metric_for_best_validation_round(split_id, "accuracy")
            except Exception:
                test_acc = "n/a"
        except Exception:
            val_acc = "n/a"
            test_acc = "n/a"
        perf_rows.append((dataset, str(val_acc), str(test_acc)))
    body = ["<p>", f"Training model: {training_model}<br>", f"Inference model: {inference_model}", "</p>"]
    body.append("<h2>Investigations</h2><ul>")
    for inv_id, dataset in investigations:
        body.append(
            f"<li><a href='../../dataset/{dataset}/investigation/{inv_id}/index.html'>Investigation {inv_id} ({dataset})</a></li>"
        )
    body.append("</ul>")
    if not perf_rows:
        raise RuntimeError("no investigation performance data found")
    body.append("<h2>Performance</h2><table border='1'>")
    body.append("<tr><th>Dataset</th><th>Validation accuracy</th><th>Test accuracy</th></tr>")
    for d, v, t in perf_rows:
        body.append(f"<tr><td>{d}</td><td>{v}</td><td>{t}</td></tr>")
    body.append("</table>")
    write_page(os.path.join(out_dir, "index.html"), f"Model {model}", "\n".join(body))


def generate_lexicostatistics_page(conn, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT lm.vendor, lm.training_model, lm.release_date,
               l.prompt_herdan, l.prompt_zipf,
               l.reasoning_herdan, l.reasoning_zipf
          FROM lexicostatistics l
          JOIN language_models lm ON l.training_model = lm.training_model
         WHERE lm.release_date IS NOT NULL
         ORDER BY lm.release_date
        """
    )
    rows = cur.fetchall()
    df = pd.DataFrame(
        rows,
        columns=[
            "Vendor",
            "Model",
            "Release Date",
            "Prompt Herdan",
            "Prompt Zipf",
            "Reasoning Herdan",
            "Reasoning Zipf",
        ],
    )

    body = ["<h2>Language Models</h2>"]
    body.append("<table border='1'>")
    body.append(
        "<tr><th>Release Date</th><th>Vendor</th><th>Model</th>"
        "<th>Prompt Herdan</th><th>Prompt Zipf</th>"
        "<th>Reasoning Herdan</th><th>Reasoning Zipf</th></tr>"
    )
    for vendor, model, date, p_h, p_z, r_h, r_z in rows:
        body.append(
            f"<tr><td>{date}</td><td>{vendor}</td><td>{model}</td>"
            f"<td>{p_h:.3f}</td><td>{p_z:.3f}</td>"
            f"<td>{r_h:.3f}</td><td>{r_z:.3f}</td></tr>"
        )
    body.append("</table>")

    if df.empty:
        raise RuntimeError("no lexicostatistics data found")
    df["Release Date"] = pd.to_datetime(df["Release Date"])
    for col, fname in [
            ("Prompt Herdan", "prompt_herdan.png"),
            ("Prompt Zipf", "prompt_zipf.png"),
            ("Reasoning Herdan", "reasoning_herdan.png"),
            ("Reasoning Zipf", "reasoning_zipf.png"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(df["Release Date"], df[col], marker="o")
            law = "Herdan" if "Herdan" in col else "Zipf"
            section = "prompts" if "Prompt" in col else "reasoning"
            ax.set_title(f"{law}'s Law Coefficients over Time ({section})")
            x_num = mdates.date2num(df["Release Date"])
            y = df[col]
            x0 = x_num.min()
            slope, intercept, r, pval, std = linregress(x_num - x0, y)
            end_date = datetime(2027, 1, 1)
            x_end = mdates.date2num(end_date)
            xs = np.linspace(x0, x_end, 100)
            ax.plot(
                mdates.num2date(xs),
                intercept + slope * (xs - x0),
                "--",
            )
            ax.set_xlim(df["Release Date"].min(), end_date)
            if law == "Herdan":
                lines = {
                    0.5: "Children's speech (~0.5–0.6)",
                    0.6: "High-school essays (~0.6–0.7)",
                    0.7: "General fiction/technical (~0.7–0.8)",
                    0.8: "Shakespeare plays (~0.8–0.9)",
                    0.9: "Highly rich vocabulary",
                }
            else:
                lines = {
                    0.7: "Academic texts (~0.7–0.9)",
                    0.8: "Shakespeare/poetry (~0.8–0.9)",
                    0.9: "Fiction/chat (~0.9–1.1)",
                    0.95: "High-school writing (~0.95–1.1)",
                    1.0: "Upper fiction bound (~1.0)",
                    1.1: "Conversation upper (~1.1)",
                }
            for y_line, label in lines.items():
                ax.axhline(y_line, color="gray", linestyle=":", linewidth=0.5)
                ax.text(
                    end_date,
                    y_line,
                    f" {label}",
                    va="bottom",
                    ha="left",
                    fontsize=8,
                    color="gray",
                )
            ax.set_xlabel("Release date")
            ax.set_ylabel(f"{law} coefficient")
            plt.xticks(rotation=45)
            fig.tight_layout()
            chart_path = os.path.join(out_dir, fname)
            plt.savefig(chart_path)
            plt.close(fig)
            body.append(f"<h2>{law} Coefficient ({section.capitalize()})</h2>")
            alt_title = f"{law}'s Law ({section})"
            body.append(f"<img src='{fname}' alt='{alt_title} over time'>")
            body.append(
                f"<p>Slope {slope:.4f}, intercept {intercept:.4f}, p={pval:.3g}</p>"
            )

    # Ensemble trends
    cur.execute(
        """
        SELECT DISTINCT ON (release_date) release_date, models
          FROM ensemble_results
         ORDER BY release_date, test_accuracy DESC
        """
    )
    ens_rows = cur.fetchall()
    if not ens_rows:
        raise RuntimeError("no ensemble trend data found")
    cur.execute(
        "SELECT training_model, prompt_herdan, prompt_zipf, reasoning_herdan, reasoning_zipf FROM lexicostatistics WHERE training_model = ANY(%s)",
        ([r[1] for r in ens_rows],),
    )
    lookup = {m: (ph, pz, rh, rz) for m, ph, pz, rh, rz in cur.fetchall()}
    data = []
    for date, models in ens_rows:
        if models in lookup:
            ph, pz, rh, rz = lookup[models]
            data.append((date, models, ph, pz, rh, rz))
    if not data:
        raise RuntimeError("no ensemble trend data found")
    ens_df = pd.DataFrame(
        data,
        columns=[
            "Release Date",
            "Models",
            "Prompt Herdan",
            "Prompt Zipf",
            "Reasoning Herdan",
            "Reasoning Zipf",
        ],
    )
    body.append("<h2>Best Ensembles</h2><table border='1'>")
    body.append(
        "<tr><th>Date</th><th>Ensemble</th>"
        "<th>Prompt Herdan</th><th>Prompt Zipf</th>"
        "<th>Reasoning Herdan</th><th>Reasoning Zipf</th></tr>"
    )
    for d_, m_, ph, pz, rh, rz in data:
        body.append(
            f"<tr><td>{d_}</td><td>{html.escape(m_)}</td><td>{ph:.3f}</td><td>{pz:.3f}</td><td>{rh:.3f}</td><td>{rz:.3f}</td></tr>"
        )
    body.append("</table>")

    ens_df["Release Date"] = pd.to_datetime(ens_df["Release Date"])
    for col, fname, title, section in [
                ("Prompt Herdan", "ensemble_prompt_herdan.png", "Herdan", "prompts"),
                ("Prompt Zipf", "ensemble_prompt_zipf.png", "Zipf", "prompts"),
                ("Reasoning Herdan", "ensemble_reasoning_herdan.png", "Herdan", "reasoning"),
                ("Reasoning Zipf", "ensemble_reasoning_zipf.png", "Zipf", "reasoning"),
            ]:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(ens_df["Release Date"], ens_df[col], marker="o")
                law_title = "Herdan's" if title == "Herdan" else "Zipf's"
                ax.set_title(f"{law_title} Law Coefficients over Time ({section})")
                x_num = mdates.date2num(ens_df["Release Date"])
                y = ens_df[col]
                x0 = x_num.min()
                slope, intercept, r, pval, std = linregress(x_num - x0, y)
                end_date = datetime(2027, 1, 1)
                x_end = mdates.date2num(end_date)
                xs = np.linspace(x0, x_end, 100)
                ax.plot(mdates.num2date(xs), intercept + slope * (xs - x0), "--")
                ax.set_xlim(ens_df["Release Date"].min(), end_date)
                if title == "Herdan":
                    lines = {
                        0.5: "Children's speech (~0.5–0.6)",
                        0.6: "High-school essays (~0.6–0.7)",
                        0.7: "General fiction/technical (~0.7–0.8)",
                        0.8: "Shakespeare plays (~0.8–0.9)",
                        0.9: "Highly rich vocabulary",
                    }
                else:
                    lines = {
                        0.7: "Academic texts (~0.7–0.9)",
                        0.8: "Shakespeare/poetry (~0.8–0.9)",
                        0.9: "Fiction/chat (~0.9–1.1)",
                        0.95: "High-school writing (~0.95–1.1)",
                        1.0: "Upper fiction bound (~1.0)",
                        1.1: "Conversation upper (~1.1)",
                    }
                for y_line, label in lines.items():
                    ax.axhline(y_line, color="gray", linestyle=":", linewidth=0.5)
                    ax.text(
                        end_date,
                        y_line,
                        f" {label}",
                        va="bottom",
                        ha="left",
                        fontsize=8,
                        color="gray",
                    )
                ax.set_xlabel("Release date")
                ax.set_ylabel(f"{title} coefficient")
                plt.xticks(rotation=45)
                fig.tight_layout()
                chart_path = os.path.join(out_dir, fname)
                plt.savefig(chart_path)
                plt.close(fig)
                body.append(f"<h2>{title} Coefficient (Ensembles - {section.capitalize()})</h2>")
                alt_title = f"{law_title} Law ({section})"
                body.append(f"<img src='{fname}' alt='{alt_title} over time'>")
                body.append(
                    f"<p>Slope {slope:.4f}, intercept {intercept:.4f}, p={pval:.3g}</p>"
                )

    write_page(os.path.join(out_dir, "index.html"), "Lexicostatistics", "\n".join(body))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export investigation results as static HTML")
    parser.add_argument("--progress-bar", action="store_true", help="show progress bars")
    args = parser.parse_args()

    base_dir = "website"
    os.makedirs(base_dir, exist_ok=True)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT dataset, config_file FROM datasets ORDER BY dataset")
    datasets = cur.fetchall()
    dataset_lookup = {d: c for d, c in datasets}
    dataset_links = []
    dataset_iter = datasets
    if args.progress_bar:
        import tqdm
        dataset_iter = tqdm.tqdm(datasets, desc="datasets")
    for dataset, cfg_file in dataset_iter:
        dataset_dir = os.path.join(base_dir, "dataset", dataset)
        generate_dataset_page(conn, dataset, cfg_file, dataset_dir, args.progress_bar)
        dataset_links.append(f"<li><a href='dataset/{dataset}/index.html'>{dataset}</a></li>")

    cur.execute(
        """
        SELECT lm.vendor, m.training_model, m.model, m.example_count
          FROM models m
          JOIN language_models lm ON m.training_model = lm.training_model
         ORDER BY lm.vendor, m.training_model, m.example_count, m.model
        """
    )
    rows = cur.fetchall()

    index_body = "<h2>Datasets</h2><ul>" + "".join(dataset_links) + "</ul>"
    index_body += "<p><a href='lexicostatistics/index.html'>Lexicostatistics</a></p>"
    index_body += "<h2>Models</h2>"

    # group models by (vendor, training_model)
    table: dict[tuple[str, str], dict[int, list[str]]] = {}
    for vendor, training_model, model, ex in rows:
        table.setdefault((vendor, training_model), {3: [], 10: []})
        if ex in (3, 10):
            table[(vendor, training_model)][ex].append(model)
        else:
            table[(vendor, training_model)].setdefault(ex, []).append(model)
        model_dir = os.path.join(base_dir, "model", model)
        generate_model_page(conn, model, model_dir, dataset_lookup)

    index_body += (
        "<table border='1'>\n"
        "<tr><th>Vendor</th><th>Language model</th><th>examples=3" +
        "</th><th>examples=10</th></tr>"
    )
    for (vendor, training_model), counts in table.items():
        ex3 = "<br>".join(
            f"<a href='model/{m}/index.html'>{m}</a>" for m in counts.get(3, [])
        )
        ex10 = "<br>".join(
            f"<a href='model/{m}/index.html'>{m}</a>" for m in counts.get(10, [])
        )
        index_body += (
            f"<tr><td>{vendor}</td><td>{training_model}</td>"
            f"<td>{ex3}</td><td>{ex10}</td></tr>"
        )
    index_body += "</table>"

    # generate lexicostatistics page
    generate_lexicostatistics_page(conn, os.path.join(base_dir, "lexicostatistics"))

    write_page(os.path.join(base_dir, "index.html"), "Narrative Learning", index_body)
    conn.close()


if __name__ == "__main__":
    main()
