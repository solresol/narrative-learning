#!/usr/bin/env python3
"""
Utility functions for generating and formatting charts in the narrative-learning project.
"""

import math
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from modules.metrics import accuracy_to_kt


def draw_baselines(
    ax,
    df,
    xpos=None,
    dataset_size=None,
    debug=None,
    best_only: bool = False,
    trend_improving: bool = True,
):
    """Draw baseline model performance as horizontal lines on a plot.

    Args:
        ax: Matplotlib axes object to draw on
        df: DataFrame containing baseline columns
        xpos: If provided, the maximum x-position used to calculate label
            placement when ``best_only`` is ``False``.  When ``None``, the
            function will use the current axis limits to determine label
            spacing.
        best_only: If ``True``, draw only the best-performing baseline.
        trend_improving: When ``best_only`` is ``True``, place the label on the
            right edge of the axis if ``True`` and on the left edge otherwise.

    Returns:
        None, modifies ax in-place
    """
    colours = {
        "dummy": "orange",
        "decision trees": "green",
        "rulefit": "purple",
        "logistic regression": "teal",
        "ebm": "gray",
        "bayesian rule list": "brown",
        "corels": "pink",
    }

    names = [m for m in colours if m in df.columns]
    if not names:
        if debug is not None:
            debug.append("no baseline columns found")
        return
    scores: dict[str, float] = {}
    for model in names:
        score = df[model].mean()
        if dataset_size is not None and not math.isnan(score):
            score = -accuracy_to_kt(score, dataset_size)
        scores[model] = score

    if best_only:
        best_model = max(scores, key=lambda m: scores[m])
        names = [best_model]

    xlim = ax.get_xlim()
    if best_only:
        x_positions = [xlim[1] if trend_improving else xlim[0]]
    else:
        if xpos is None:
            start, end = xlim
            convert_dates = False
        else:
            start, end = xlim[0], xpos
            convert_dates = isinstance(end, (datetime, pd.Timestamp, np.datetime64))

        if convert_dates:
            end = mdates.date2num(end)
        x_positions = np.linspace(start, end, len(names))
        if convert_dates:
            x_positions = mdates.num2date(x_positions)

    for model, xpos_val in zip(names, x_positions):
        score = scores[model]
        ax.axhline(score, linestyle="dotted", c=colours[model])
        if debug is not None:
            debug.append(f"axhline {model} at {score:.3f}")
        if best_only:
            offset = -5 if trend_improving else 5
            ha = "right" if trend_improving else "left"
            ax.annotate(
                xy=(xpos_val, score - 0.03),
                text=model.title(),
                c=colours[model],
                ha=ha,
                xytext=(offset, 0),
                textcoords="offset points",
            )
            if debug is not None:
                debug.append(f"annotate {model} at edge {xpos_val}")
        else:
            ax.annotate(
                xy=(xpos_val, score - 0.03),
                text=model.title(),
                c=colours[model],
                ha="center",
            )
            if debug is not None:
                debug.append(f"annotate {model} at {xpos_val}")
