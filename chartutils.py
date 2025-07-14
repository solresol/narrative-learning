#!/usr/bin/env python3
"""
Utility functions for generating and formatting charts in the narrative-learning project.
"""

import math
import matplotlib.pyplot as plt
from modules.metrics import accuracy_to_kt


def draw_baselines(ax, df, xpos=12.5, dataset_size=None):
    """Draw baseline model performance as horizontal lines on a plot.
    
    Args:
        ax: Matplotlib axes object to draw on
        df: DataFrame containing baseline columns
        xpos: X-position for annotation labels (default: 12.5)
    
    Returns:
        None, modifies ax in-place
    """
    colours = {
        'logistic regression': 'teal',
        'decision trees': 'gold',
        'dummy': 'orange',
        'rulefit': 'purple',
        'bayesian rule list': 'brown',
        'corels': 'pink',
        'ebm': 'gray',
    }
    
    for model, colour in colours.items():
        if model in df.columns:
            # Don't need to take the mean -- they will all be the same value
            score = df[model].mean()
            if dataset_size is not None and not math.isnan(score):
                score = -accuracy_to_kt(score, dataset_size)
            ax.axhline(score, linestyle='dotted', c=colours[model])
            ax.annotate(xy=(xpos, score-0.03), text=model.title(), c=colours[model])
