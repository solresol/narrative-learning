#!/usr/bin/env python3
"""Generate a static HTML representation of the investigations database."""
from __future__ import annotations
import os
import html
import argparse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Tuple, Optional
import subprocess
import tempfile

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress, wilcoxon
import numpy as np
import umap

import pandas as pd

from modules.postgres import get_connection, get_investigation_settings
from modules.results_loader import load_results_dataframe
from datasetconfig import DatasetConfig
from chartutils import draw_baselines
from modules.ensemble_selection import get_interesting_ensembles
from modules.metrics import accuracy_to_kt
from modules.investigation_status import lookup_incomplete_investigations
import math

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(["html", "xml"]),
)
_page_template = _env.get_template("page.html")


def write_page(path: str, title: str, body: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    html_content = _page_template.render(title=title, body=body)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)


def get_split_id(cfg: DatasetConfig) -> int:
    cur = cfg.conn.cursor()
    cfg._execute(cur, f"SELECT MIN(split_id) FROM {cfg.splits_table}")
    row = cur.fetchone()
    return row[0]


def plot_release_chart(
    conn, dataset: str, df: pd.DataFrame, out_path: str, dataset_size: int
) -> tuple[Optional[tuple[float, float, float]], list[str]]:
    """Plot test scores by model release date for a dataset.

    Returns the slope, intercept and p-value of the regression line calculated
    over the ensemble maximum scores, or ``None`` if no ensemble data exists.
    """
    actions: list[str] = []
    if df.empty:
        return None, actions
    cur = conn.cursor()
    cur.execute(
        "SELECT training_model, release_date, ollama_hosted FROM language_models"
    )
    info = pd.DataFrame(cur.fetchall(), columns=["Model", "Release Date", "ollama"])
    df = df.merge(info, on="Model", how="left")
    actions.append(f"merge info: {len(df)} rows")
    # ``ollama`` is True for models hosted on Ollama. These rows should be
    # excluded from the plot, treating missing values as False.
    df = df[~df["ollama"].fillna(False).astype(bool)]
    actions.append(f"filter ollama: {len(df)} rows")
    df.dropna(subset=["Release Date", "Accuracy"], inplace=True)
    actions.append(f"drop NA release/accuracy: {len(df)} rows")
    if df.empty:
        raise RuntimeError("no model results found to plot")

    df["Release Date"] = pd.to_datetime(df["Release Date"], utc=True)
    df.sort_values("Release Date", inplace=True)
    df["KT"] = df["Accuracy"].apply(lambda x: accuracy_to_kt(x, dataset_size))
    actions.append("compute KT and sort")

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
    best_baseline_y = None
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
        baseline_vals = []
        for col_db, col_df, val in zip(cols_db, cols_df, row):
            df[col_df] = val
            if val is not None:
                baseline_vals.append(-accuracy_to_kt(val, dataset_size))
        if baseline_vals:
            best_baseline_y = max(baseline_vals)
    actions.append("added baselines")

    ens_df = get_interesting_ensembles(conn, dataset)
    if ens_df.empty:
        raise RuntimeError("no interesting ensembles found")

    fig, ax = plt.subplots(figsize=(8, 4))
    actions.append("create figure")
    ax.scatter(df["Release Date"], -df["KT"], marker="o", label="model")
    coords = list(zip(df["Release Date"].dt.date, (-df["KT"]).tolist()))
    actions.append(f"scatter models: {coords}")
    ax.set_xlabel("Model release date")
    ax.set_ylabel("-log10 KT accuracy")
    draw_baselines(
        ax,
        df,
        xpos=df["Release Date"].max(),
        dataset_size=dataset_size,
        debug=actions,
    )

    ens_df = ens_df.copy()
    ens_df["Release Date"] = pd.to_datetime(ens_df["release_date"], utc=True)
    ens_df.sort_values("Release Date", inplace=True)
    ens_df["KT"] = (ens_df["test_correct"] / ens_df["test_total"]).apply(
        lambda x: accuracy_to_kt(x, dataset_size)
    )

    earliest_date = min(df["Release Date"].min(), ens_df["Release Date"].min())
    xlim_min = earliest_date.to_pydatetime().replace(day=1)
    actions.append(f"xlim_min {xlim_min.date()}")

    latest_model = df["Release Date"].max()
    latest_ens = ens_df["Release Date"].max()
    ax.scatter(
        ens_df["Release Date"],
        -ens_df["KT"],
        marker="x",
        c="red",
        label="ensemble",
    )
    actions.append(f"scatter ensembles: {len(ens_df)}")
    x = mdates.date2num(ens_df["Release Date"])
    y = -ens_df["KT"]
    xlim_max_candidate = max(latest_model, latest_ens)
    cross_date = None
    if len(ens_df) > 1:
        slope, intercept, r, pval, std = linregress(x, y)
        if best_baseline_y is not None and slope > 0:
            cross_x = (best_baseline_y - intercept) / slope
            cross_date = mdates.num2date(cross_x, tz=timezone.utc)
            future_limit = datetime.now(timezone.utc) + relativedelta(months=18)
            if cross_date <= future_limit:
                xlim_max_candidate = max(xlim_max_candidate, cross_date)
        xs = np.linspace(mdates.date2num(xlim_min), mdates.date2num(xlim_max_candidate), 100)
        ax.plot(mdates.num2date(xs), intercept + slope * xs, "--", c="red")
        actions.append("draw ensemble trend line")
    else:
        slope = intercept = pval = float("nan")

    xlim_max = xlim_max_candidate
    ax.set_xlim(xlim_min, xlim_max)
    actions.append(f"xlim_max {xlim_max.date()}")

    if cross_date is not None and xlim_min <= cross_date <= xlim_max:
        ax.axvline(cross_date, linestyle=":", color="gray")
        ax.annotate(
            cross_date.strftime("%Y-%m-%d"),
            xy=(cross_date, best_baseline_y),
            xytext=(0, 5),
            textcoords="offset points",
            rotation=90,
            ha="center",
            va="bottom",
        )
        actions.append("mark baseline crossing")

    ax.set_title(f"{dataset} test scores by release date")

    ax.legend()
    fig.tight_layout()
    plt.savefig(out_path)
    actions.append(f"savefig {out_path}")
    plt.close(fig)
    return (slope, intercept, pval), actions


def plot_feature_scatter(cfg: DatasetConfig, out_path: str) -> bool:
    """Plot a 2D scatter of dataset features.

    If the dataset has exactly two feature columns, they are used directly.
    Otherwise the features are reduced to two dimensions using UMAP with a
    fixed random state so the plot is deterministic.
    """
    features = [
        c
        for c in cfg.columns
        if c not in (cfg.primary_key, cfg.target_field)
    ]
    if len(features) < 2:
        return False

    cur = cfg.conn.cursor()
    cols = ", ".join([cfg._ident(c) for c in features + [cfg.target_field]])
    cfg._execute(cur, f"SELECT {cols} FROM {cfg.table_name}")
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=features + [cfg.target_field])

    if len(features) > 2:
        # Convert categorical data to numeric via one-hot encoding
        X = pd.get_dummies(df[features])
        embedding = umap.UMAP(n_components=2, random_state=0).fit_transform(X)
        scatter_df = pd.DataFrame(embedding, columns=["x", "y"])
    else:
        scatter_df = df[features].rename(columns={features[0]: "x", features[1]: "y"})

    scatter_df[cfg.target_field] = df[cfg.target_field]

    markers = ["o", "s", "^", "v", "D", "X", "P", "<", ">", "*"]
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, target_val in enumerate(sorted(scatter_df[cfg.target_field].unique())):
        mask = scatter_df[cfg.target_field] == target_val
        ax.scatter(
            scatter_df.loc[mask, "x"],
            scatter_df.loc[mask, "y"],
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            label=str(target_val),
        )

    ax.set_xlabel(features[0] if len(features) == 2 else "UMAP1")
    ax.set_ylabel(features[1] if len(features) == 2 else "UMAP2")
    ax.legend()
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    return True


def generate_round_page(
    cfg: DatasetConfig, investigation_id: int, round_id: int, out_dir: str
) -> None:
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
    body.append(
        f"<li>Train accuracy: {train_acc if train_acc is not None else 'n/a'}</li>"
    )
    body.append(
        f"<li>Validation accuracy: {val_acc if val_acc is not None else 'n/a'}</li>"
    )
    body.append(
        f"<li>Test accuracy: {test_acc if test_acc is not None else 'n/a'}</li>"
    )
    body.append("</ul>")
    write_page(
        os.path.join(out_dir, "index.html"), f"Round {round_id}", "\n".join(body)
    )


def generate_investigation_page(
    conn,
    dataset: str,
    cfg_file: str,
    inv_id: int,
    base_dir: str,
    status: str | None = None,
) -> int | None:
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
    if status:
        body.append(f"<li><b>Status:</b> {html.escape(status)}</li>")
    body.append(f"<li>Dataset: {dataset}</li>")
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
        "<tr><th>Round</th><th>UUID</th><th>Started</th><th>Completed"
        + "</th><th>Val Acc</th><th>Val KT</th><th>Test Acc</th><th>Test KT</th></tr>"
    )
    for (
        r_id,
        r_uuid,
        r_start,
        r_completed,
        train_acc,
        v_acc,
        t_acc,
        inf_count,
    ) in rounds:
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
        generate_round_page(
            cfg, inv_id, r_id, os.path.join(inv_dir, "round", str(r_id))
        )
    body.append("</table>")

    if df_rounds.empty:
        raise RuntimeError(f"no rounds data found for investigation {inv_id}")
    plot_df = (
        df_rounds.dropna(subset=["val_acc"])
        .sort_values("round_start")
        .reset_index(drop=True)
    )
    plot_df["rank"] = plot_df.index + 1
    best_rank = int(plot_df["val_acc"].idxmax()) + 1 if not plot_df.empty else None
    for col in ["train_acc", "val_acc", "test_acc"]:
        plot_df[col] = plot_df[col].apply(lambda x: -accuracy_to_kt(x, dataset_size))
    plt.figure(figsize=(8, 4))
    plt.plot(plot_df["rank"], plot_df["train_acc"], label="train")
    plt.plot(plot_df["rank"], plot_df["val_acc"], label="validation")
    plt.plot(plot_df["rank"], plot_df["test_acc"], label="test")
    if len(plot_df) > 20:
        ticks = plot_df["rank"]
        labels = [
            r_id if (i + 1) % 5 == 0 else ""
            for i, r_id in enumerate(plot_df["round_id"])
        ]
        plt.xticks(ticks, labels)
    else:
        plt.xticks(plot_df["rank"], plot_df["round_id"])
    plt.xlabel("Round")
    plt.ylabel("Negative log10 KT score")
    plt.title("Round Scores")
    plt.legend()
    plt.tight_layout()
    chart_path = os.path.join(inv_dir, "scores.png")
    plt.savefig(chart_path)
    plt.close()
    body.append("<h2>Scores</h2>")
    body.append(f"<img src='scores.png' alt='round scores'>")

    write_page(
        os.path.join(inv_dir, "index.html"), f"Investigation {inv_id}", "\n".join(body)
    )

    return best_rank


def generate_dataset_page(
    conn,
    dataset: str,
    cfg_file: str,
    out_dir: str,
    short_summary: str | None = None,
    incomplete: dict[int, str] | None = None,
) -> tuple[float, float, float] | None:
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

    scatter_path = os.path.join(out_dir, "feature_scatter.png")
    if plot_feature_scatter(cfg, scatter_path):
        body = [
            "<h2>Feature Scatter</h2>",
            "<img src='feature_scatter.png' alt='feature scatter'>",
        ]
    else:
        body = []
    if short_summary:
        body.append(f"<p>{html.escape(short_summary)}</p>")
    cur.execute(
        "SELECT provenance FROM dataset_provenance WHERE dataset = %s",
        (dataset,),
    )
    row = cur.fetchone()
    if row:
        body.append("<h2>Provenance</h2>")
        body.append(f"<pre>{html.escape(row[0])}</pre>")

    best_ranks: list[int] = []
    for inv_id, model, _training_model in investigations:
        reason = incomplete.get(inv_id) if incomplete else None
        rank = generate_investigation_page(
            conn, dataset, cfg_file, inv_id, out_dir, reason
        )
        if rank is not None:
            best_ranks.append(rank)

    if best_ranks:
        plt.figure(figsize=(6, 4))
        bins = range(1, max(best_ranks) + 2)
        plt.hist(best_ranks, bins=bins, edgecolor="black", align="left")
        plt.xlabel("number of rounds")
        plt.ylabel("frequency")
        plt.title("Best validation round position")
        plt.tight_layout()
        rank_chart = os.path.join(out_dir, "best_round_rank.png")
        plt.savefig(rank_chart)
        plt.close()
        body.append("<h2>Best Validation Round Rank</h2>")
        body.append("<img src='best_round_rank.png' alt='best round rank distribution'>")
    else:
        raise RuntimeError("no round rank data found")

    # Load results directly from the database
    df_results = load_results_dataframe(conn, dataset, cfg_file)

    # Plot test results by model release date
    chart_file = os.path.join(out_dir, f"release_scores_{dataset}.png")
    stats, debug_actions = plot_release_chart(
        conn, dataset, df_results, chart_file, dataset_size
    )
    if os.path.exists(chart_file):
        body.append("<h2>Test scores by release date</h2>")
        body.append(
            f"<img src='release_scores_{dataset}.png' alt='scores by release date'>"
        )
        if stats is not None:
            slope, intercept, pval = stats
            body.append(
                f"<p>Regression slope: {slope:.4f}, intercept: {intercept:.4f}, p-value: {pval:.5g}</p>"
            )
        if debug_actions:
            body.append("<h3>Debug</h3>")
            body.append("<pre>" + "\n".join(debug_actions) + "</pre>")

    if not df_results.empty:
        df = df_results.copy()
        cur.execute(
            "SELECT training_model, release_date, ollama_hosted FROM language_models"
        )
        info = pd.DataFrame(cur.fetchall(), columns=["Model", "Release Date", "ollama"])
        df = df.merge(info, on="Model", how="left")
        df = df[~df["ollama"].fillna(False).astype(bool)]
        df.dropna(subset=["Release Date", "Accuracy"], inplace=True)
        df["Release Date"] = pd.to_datetime(df["Release Date"], utc=True)
        df.sort_values("Release Date", inplace=True)
        df["KT"] = df["Accuracy"].apply(lambda x: accuracy_to_kt(x, dataset_size))
        body.append("<h2>Model Scores</h2><table border='1'>")
        body.append(
            "<tr><th>Model</th><th>Run Name</th><th>Investigation</th><th>Release Date"
            + "</th><th>Examples</th><th>Patience</th><th>Rounds</th><th>Val Acc"
            + "</th><th>Val KT</th><th>Test Acc</th><th>Test KT</th></tr>"
        )
        for m, d_, run_name, inv_id, ex, patience in df[
            [
                "Model",
                "Release Date",
                "Run Name",
                "Investigation",
                "Sampler",
                "Patience",
            ]
        ].itertuples(index=False):
            cfg = DatasetConfig(conn, cfg_file, dataset, inv_id)
            split_id = cfg.get_latest_split_id()
            round_count = len(cfg.get_processed_rounds_for_split(split_id))
            try:
                best_round = cfg.get_best_round_id(split_id, "accuracy")
                val_df = cfg.generate_metrics_data(split_id, "accuracy", "validation")
                val_acc = val_df[val_df.round_id == best_round].metric.iloc[0]
                test_acc = cfg.get_test_metric_for_best_validation_round(
                    split_id, "accuracy"
                )
            except Exception:
                val_acc = test_acc = None
            val_acc_disp = f"{val_acc:.3f}" if val_acc is not None else "n/a"
            val_kt = (
                f"{accuracy_to_kt(val_acc, dataset_size):.3f}"
                if val_acc is not None
                else "n/a"
            )
            test_acc_disp = f"{test_acc:.3f}" if test_acc is not None else "n/a"
            test_kt = (
                f"{accuracy_to_kt(test_acc, dataset_size):.3f}"
                if test_acc is not None
                else "n/a"
            )
            label = str(inv_id)
            if incomplete and inv_id in incomplete:
                label += f" ({html.escape(incomplete[inv_id])})"
            body.append(
                f"<tr><td>{m}</td><td>{run_name.rstrip('.')}</td>"
                f"<td><a href='investigation/{inv_id}/index.html'>{label}</a></td>"
                f"<td>{d_.date()}</td><td>{ex}</td><td>{patience}</td><td>{round_count}</td>"
                f"<td><a href='investigation/{inv_id}/round/{best_round}/index.html'>{val_acc_disp}</a></td>"
                f"<td>{val_kt}</td><td>{test_acc_disp}</td><td>{test_kt}</td></tr>"
            )
        body.append("</table>")

        ens_df = get_interesting_ensembles(conn, dataset)
        if ens_df.empty:
            raise RuntimeError("no ensemble data found")
        ens_df = ens_df.copy()
        ens_df["Release Date"] = pd.to_datetime(ens_df["release_date"], utc=True)
        ens_df.sort_values("Release Date", inplace=True)
        ens_df["validation_kt"] = ens_df["validation_accuracy"].apply(
            lambda x: accuracy_to_kt(x, dataset_size)
        )
        ens_df["test_accuracy"] = ens_df["test_correct"] / ens_df["test_total"]
        ens_df["test_kt"] = ens_df["test_accuracy"].apply(
            lambda x: accuracy_to_kt(x, dataset_size)
        )
        body.append("<h2>Ensemble Max Scores</h2><table border='1'>")
        body.append(
            "<tr><th>Release Date</th><th>Val Acc</th><th>Val KT</th><th>Test Acc</th><th>Test KT</th><th>Ensemble</th></tr>"
        )
        for d_, v_acc, v_kt, t_acc, t_kt, names in ens_df[
            [
                "Release Date",
                "validation_accuracy",
                "validation_kt",
                "test_accuracy",
                "test_kt",
                "model_names",
            ]
        ].itertuples(index=False):
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
            body.append(
                f"<tr><td>{name}</td><td>{display_acc}</td><td>{display_kt}</td></tr>"
            )
        body.append("</table>")

        body.append("<h2>Baseline Models</h2>")

        cur.execute(
            "SELECT feature, weight FROM baseline_logreg WHERE dataset = %s ORDER BY feature",
            (dataset,),
        )
        rows = cur.fetchall()
        if rows:
            body.append("<h3>Logistic Regression</h3>")
            body.append("<table border='1'><tr><th>Feature</th><th>Weight</th></tr>")
            instructions = []
            intercept = None
            for feat, wt in rows:
                body.append(f"<tr><td>{html.escape(feat)}</td><td>{wt:.3f}</td></tr>")
                if feat == "intercept":
                    intercept = wt
                else:
                    instructions.append(
                        f"take the {html.escape(feat)} and multiply by {wt:.3f}"
                    )
            body.append("</table>")
            if intercept is not None and instructions:
                pos = html.escape(cfg.positive_label())
                neg = html.escape(cfg.negative_label())
                instr_text = ", ".join(instructions)
                body.append(
                    f"<p>{instr_text}, add those up and if the number is greater than {intercept:.3f} then predict {pos}, otherwise predict {neg}.</p>"
                )

        cur.execute(
            "SELECT dot_data FROM baseline_decision_tree WHERE dataset = %s",
            (dataset,),
        )
        row = cur.fetchone()
        if row and row[0]:
            body.append("<h3>Decision Tree</h3>")
            dot_data = row[0]
            img_name = "decision_tree.png"
            img_path = os.path.join(out_dir, img_name)
            dot_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w", suffix=".dot", delete=False
                ) as tf:
                    tf.write(dot_data)
                    dot_path = tf.name
                subprocess.run(
                    [
                        "dot",
                        "-Tpng",
                        dot_path,
                        "-o",
                        img_path,
                    ],
                    check=True,
                    timeout=30,
                    capture_output=True,
                    text=True,
                )
                body.append(f"<img src='{img_name}' alt='decision tree'>")
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                body.append(f"<pre class='graphviz'>{html.escape(dot_data)}</pre>")
                err = getattr(e, "stderr", None)
                if err:
                    body.append(f"<pre>{html.escape(str(err))}</pre>")
                else:
                    body.append(f"<pre>{html.escape(str(e))}</pre>")
            finally:
                if dot_path and os.path.exists(dot_path):
                    os.remove(dot_path)

        cur.execute(
            "SELECT constant_value FROM baseline_dummy WHERE dataset = %s",
            (dataset,),
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            body.append("<h3>Dummy</h3>")
            body.append(f"<p>Always predict {html.escape(str(row[0]))}</p>")

        cur.execute(
            "SELECT rule_index, rule, weight FROM baseline_rulefit WHERE dataset = %s ORDER BY rule_index",
            (dataset,),
        )
        rows = cur.fetchall()
        if rows:
            body.append("<h3>RuleFit</h3>")
            body.append(
                "<table border='1'><tr><th>#</th><th>Rule</th><th>Weight</th></tr>"
            )
            for idx, rule, wt in rows:
                body.append(
                    f"<tr><td>{idx}</td><td>{html.escape(rule)}</td><td>{wt:.3f}</td></tr>"
                )
            body.append("</table>")

        cur.execute(
            "SELECT rule_order, rule, probability FROM baseline_bayesian_rule_list WHERE dataset = %s ORDER BY rule_order",
            (dataset,),
        )
        rows = cur.fetchall()
        if rows:
            body.append("<h3>Bayesian Rule List</h3>")
            body.append(
                "<table border='1'><tr><th>#</th><th>Rule</th><th>Probability</th></tr>"
            )
            for idx, rule, prob in rows:
                prob_disp = f"{prob:.3f}" if prob is not None else "n/a"
                body.append(
                    f"<tr><td>{idx}</td><td>{html.escape(rule)}</td><td>{prob_disp}</td></tr>"
                )
            body.append("</table>")

        cur.execute(
            "SELECT rule_order, rule FROM baseline_corels WHERE dataset = %s ORDER BY rule_order",
            (dataset,),
        )
        rows = cur.fetchall()
        if rows:
            body.append("<h3>CORELS</h3>")
            body.append("<ol>")
            for _, rule in rows:
                body.append(f"<li>{html.escape(rule)}</li>")
            body.append("</ol>")

        cur.execute(
            "SELECT feature, contributions FROM baseline_ebm WHERE dataset = %s ORDER BY feature",
            (dataset,),
        )
        rows = cur.fetchall()
        if rows:
            body.append("<h3>EBM</h3>")
            body.append(
                "<table border='1'><tr><th>Feature</th><th>Contribution Data</th></tr>"
            )
            for feat, contrib in rows:
                body.append(
                    f"<tr><td>{html.escape(feat)}</td><td>{html.escape(str(contrib))}</td></tr>"
                )
            body.append("</table>")

    write_page(
        os.path.join(out_dir, "index.html"), f"Dataset {dataset}", "\n".join(body)
    )

    return stats


def generate_dataset_index_page(
    dataset_rows: List[tuple[str, str | None]], out_path: str
) -> None:
    body = ["<table border='1'>", "<tr><th>Dataset</th><th>Description</th></tr>"]
    for dataset, summary in dataset_rows:
        desc = html.escape(summary) if summary else ""
        body.append(
            f"<tr><td><a href='{dataset}/index.html'>{dataset}</a></td><td>{desc}</td></tr>"
        )
    body.append("</table>")
    write_page(out_path, "Datasets", "\n".join(body))


def generate_model_page(
    conn,
    model: str,
    out_dir: str,
    dataset_lookup: dict[str, str],
    incomplete: dict[int, str] | None = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cur = conn.cursor()
    cur.execute(
        "SELECT training_model, inference_model FROM models WHERE model = %s", (model,)
    )
    row = cur.fetchone()
    if not row:
        return
    training_model, inference_model = row
    cur.execute(
        "SELECT id, dataset FROM investigations WHERE model = %s ORDER BY id", (model,)
    )
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
                test_acc = cfg.get_test_metric_for_best_validation_round(
                    split_id, "accuracy"
                )
            except Exception:
                test_acc = "n/a"
        except Exception:
            val_acc = "n/a"
            test_acc = "n/a"
        perf_rows.append((dataset, str(val_acc), str(test_acc)))
    body = [
        "<p>",
        f"Training model: {training_model}<br>",
        f"Inference model: {inference_model}",
        "</p>",
    ]
    body.append("<h2>Investigations</h2><ul>")
    for inv_id, dataset in investigations:
        label = f"Investigation {inv_id} ({dataset})"
        if incomplete and inv_id in incomplete:
            label += f" - {html.escape(incomplete[inv_id])}"
        body.append(
            f"<li><a href='../../dataset/{dataset}/investigation/{inv_id}/index.html'>{label}</a></li>"
        )
    body.append("</ul>")
    if not perf_rows:
        raise RuntimeError(f"no investigation performance data found for {model}")
    body.append("<h2>Performance</h2><table border='1'>")
    body.append(
        "<tr><th>Dataset</th><th>Validation accuracy</th><th>Test accuracy</th></tr>"
    )
    for d, v, t in perf_rows:
        body.append(f"<tr><td>{d}</td><td>{v}</td><td>{t}</td></tr>")
    body.append("</table>")
    write_page(os.path.join(out_dir, "index.html"), f"Model {model}", "\n".join(body))


def generate_model_index_page(
    conn,
    dataset_lookup: dict[str, str],
    rows: List[tuple[str, datetime, str, str, int]],
    out_path: str,
) -> None:
    body = [
        "<table border='1'>",
        "<tr><th>Vendor</th><th>Language model</th><th>Release Date</th><th>examples=3</th><th>examples=10</th></tr>",
    ]
    table: dict[tuple[str, str, str], dict[int, list[str]]] = {}
    for vendor, release_date, training_model, model, ex in rows:
        key = (vendor, training_model, str(release_date))
        table.setdefault(key, {3: [], 10: []})
        if ex in (3, 10):
            table[key][ex].append(model)
        else:
            table[key].setdefault(ex, []).append(model)
    for (vendor, training_model, date), counts in table.items():
        ex3 = "<br>".join(
            f"<a href='{m}/index.html'>{m}</a>" for m in counts.get(3, [])
        )
        ex10 = "<br>".join(
            f"<a href='{m}/index.html'>{m}</a>" for m in counts.get(10, [])
        )
        body.append(
            f"<tr><td>{vendor}</td><td>{training_model}</td><td>{date}</td><td>{ex3}</td><td>{ex10}</td></tr>"
        )
    body.append("</table>")

    # gather performance data across datasets
    df_list = []
    for dataset, cfg_file in dataset_lookup.items():
        df = load_results_dataframe(conn, dataset, cfg_file)
        df_list.append(df)
    if df_list:
        full_df = pd.concat(df_list, ignore_index=True)
        full_df["Neg Log KT"] = full_df.apply(
            lambda r: -accuracy_to_kt(r["Accuracy"], r["Data Points"]) if r["Accuracy"] is not None else np.nan,
            axis=1,
        )
        pivot = full_df.pivot_table(
            index=["Task", "Model"], columns="Sampler", values="Neg Log KT"
        )
        plot_df = pivot.dropna(subset=[3, 10])
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(plot_df[3], plot_df[10], marker="o")
            lo = min(plot_df[3].min(), plot_df[10].min())
            hi = max(plot_df[3].max(), plot_df[10].max())
            ax.plot([lo, hi], [lo, hi], "r--")
            ax.set_xlabel("-log10 KT (examples=3)")
            ax.set_ylabel("-log10 KT (examples=10)")
            fig.tight_layout()
            scatter_file = os.path.join(os.path.dirname(out_path), "examples_scatter.png")
            plt.savefig(scatter_file)
            plt.close(fig)

            diff = plot_df[10] - plot_df[3]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(diff, bins=10, edgecolor="black")
            ax.set_xlabel("Difference in -log10 KT")
            ax.set_ylabel("Frequency")
            fig.tight_layout()
            hist_file = os.path.join(os.path.dirname(out_path), "examples_diff_hist.png")
            plt.savefig(hist_file)
            plt.close(fig)

            stat, pval = wilcoxon(plot_df[10], plot_df[3])

            body.append("<h2>Example Count Comparison</h2>")
            body.append(f"<img src='examples_scatter.png' alt='examples scatter'>")
            body.append(f"<img src='examples_diff_hist.png' alt='difference histogram'>")
            body.append(f"<p>Wilcoxon statistic: {stat:.2f}, p-value: {pval:.5g}</p>")

    write_page(out_path, "Models", "\n".join(body))


def generate_lexicostatistics_page(conn, out_dir: str) -> dict[str, tuple[float, float, float]]:
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
    rows = [
        row
        for row in cur.fetchall()
        if not any(v == 0 for v in row[3:])
    ]
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
    stats: dict[str, tuple[float, float, float]] = {}
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
    df["Release Date"] = pd.to_datetime(df["Release Date"], utc=True)
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
        # ax.set_xlim(df["Release Date"].min(), end_date)
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
            f"<p>Slope {slope:.4f}, intercept {intercept:.4f}, p={pval:.5g}</p>"
        )
        stats[fname] = (slope, intercept, pval)

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
    lookup = {
        m: (ph, pz, rh, rz)
        for m, ph, pz, rh, rz in cur.fetchall()
        if not any(v == 0 for v in (ph, pz, rh, rz))
    }
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

    ens_df["Release Date"] = pd.to_datetime(ens_df["Release Date"], utc=True)
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
        # ax.set_xlim(ens_df["Release Date"].min(), end_date)
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
        body.append(
            f"<h2>{title} Coefficient (Ensembles - {section.capitalize()})</h2>"
        )
        alt_title = f"{law_title} Law ({section})"
        body.append(f"<img src='{fname}' alt='{alt_title} over time'>")
        body.append(
            f"<p>Slope {slope:.4f}, intercept {intercept:.4f}, p={pval:.5g}</p>"
        )
        stats[fname] = (slope, intercept, pval)

    write_page(os.path.join(out_dir, "index.html"), "Lexicostatistics", "\n".join(body))

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export investigation results as static HTML"
    )
    parser.add_argument(
        "--progress-bar", action="store_true", help="show progress bars"
    )
    args = parser.parse_args()

    base_dir = "website"
    os.makedirs(base_dir, exist_ok=True)
    conn = get_connection()
    incomplete_lookup = lookup_incomplete_investigations(conn)
    cur = conn.cursor()
    cur.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='datasets' AND column_name='short_summary'"
    )
    has_summary = cur.fetchone() is not None
    cur.execute(
        f"SELECT dataset, config_file{', short_summary' if has_summary else ''} FROM datasets ORDER BY dataset"
    )
    datasets = cur.fetchall()
    dataset_lookup = {d: c for d, c, *rest in datasets}
    dataset_rows: List[tuple[str, str | None]] = []
    dataset_iter = datasets
    if args.progress_bar:
        import tqdm

        dataset_iter = tqdm.tqdm(datasets, desc="datasets")
    dataset_stats: dict[str, tuple[float, float, float] | None] = {}
    for row in dataset_iter:
        dataset, cfg_file, *rest = row
        summary = rest[0] if rest else None
        dataset_dir = os.path.join(base_dir, "dataset", dataset)
        stats = generate_dataset_page(
            conn,
            dataset,
            cfg_file,
            dataset_dir,
            summary,
            incomplete_lookup,
        )
        dataset_rows.append((dataset, summary))
        dataset_stats[dataset] = stats

    generate_dataset_index_page(
        dataset_rows, os.path.join(base_dir, "dataset", "index.html")
    )
    dataset_names = [d for d, _ in dataset_rows]

    # generate lexicostatistics page before building the index body
    lex_stats = generate_lexicostatistics_page(conn, os.path.join(base_dir, "lexicostatistics"))

    cur.execute(
        """
        SELECT lm.vendor, lm.release_date, m.training_model, m.model, m.example_count
          FROM models m
          JOIN language_models lm ON m.training_model = lm.training_model
         ORDER BY lm.vendor, m.training_model, m.example_count, m.model
        """
    )
    rows = cur.fetchall()

    for vendor, release_date, training_model, model, ex in rows:
        model_dir = os.path.join(base_dir, "model", model)
        generate_model_page(conn, model, model_dir, dataset_lookup, incomplete_lookup)

    generate_model_index_page(conn, dataset_lookup, rows, os.path.join(base_dir, "model", "index.html"))

    index_body_parts = [
        "<p>Narrative Learning studies the iterative training of reasoning models that explain their answers.</p>",
        "<p>This site serves as an observatory to track progress and compare ensembles to traditional explainable models.</p>",
    ]
    for d in dataset_names:
        index_body_parts.append(f"<h2>{d}</h2>")
        index_body_parts.append(
            f"<img src='dataset/{d}/release_scores_{d}.png' alt='ensemble accuracy trend for {d}'>"
        )
        stats = dataset_stats.get(d)
        if stats:
            s, i, p = stats
            index_body_parts.append(
                f"<p>Slope {s:.4f}, intercept {i:.4f}, p={p:.5g}</p>"
            )
    index_body_parts.append("<h2>Lexicostatistics</h2>")
    index_body_parts.append(
        "<img src='lexicostatistics/ensemble_prompt_herdan.png' alt='Prompt vocabulary trend'>"
    )
    s = lex_stats.get("ensemble_prompt_herdan.png") if lex_stats else None
    if s:
        index_body_parts.append(
            f"<p>Slope {s[0]:.4f}, intercept {s[1]:.4f}, p={s[2]:.5g}</p>"
        )
    index_body_parts.append(
        "<img src='lexicostatistics/ensemble_reasoning_herdan.png' alt='Reasoning vocabulary trend'>"
    )
    s = lex_stats.get("ensemble_reasoning_herdan.png") if lex_stats else None
    if s:
        index_body_parts.append(
            f"<p>Slope {s[0]:.4f}, intercept {s[1]:.4f}, p={s[2]:.5g}</p>"
        )
    index_body_parts.append(
        "<p><a href='dataset/index.html'>Datasets</a> | <a href='model/index.html'>Models</a> | <a href='lexicostatistics/index.html'>Lexicostatistics</a></p>"
    )
    index_body = "\n".join(index_body_parts)

    write_page(os.path.join(base_dir, "index.html"), "Narrative Learning", index_body)
    conn.close()


if __name__ == "__main__":
    main()
