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


def draw_baselines(ax, df, xpos=None, dataset_size=None):
    """Draw baseline model performance as horizontal lines on a plot.

    Args:
        ax: Matplotlib axes object to draw on
        df: DataFrame containing baseline columns
        xpos: If provided, the maximum x-position used to calculate label
            placement. When ``None``, the function will use the current axis
            limits to determine label spacing.

    Returns:
        None, modifies ax in-place
    """
    colours = {
        "logistic regression": "teal",
        "decision trees": "gold",
        "dummy": "orange",
        "rulefit": "purple",
        "bayesian rule list": "brown",
        "corels": "pink",
        "ebm": "gray",
    }

    names = [m for m in colours if m in df.columns]
    if not names:
        return

    xlim = ax.get_xlim()
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
        score = df[model].mean()
        if dataset_size is not None and not math.isnan(score):
            score = -accuracy_to_kt(score, dataset_size)
        ax.axhline(score, linestyle="dotted", c=colours[model])
        ax.annotate(
            xy=(xpos_val, score - 0.03),
            text=model.title(),
            c=colours[model],
            ha="center",
        )
