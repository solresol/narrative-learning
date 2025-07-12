#!/usr/bin/env python3
"""
Utility functions for generating and formatting charts in the narrative-learning project.
"""

import math
import matplotlib.pyplot as plt


def draw_baselines(ax, df, xpos=12.5):
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
            # Convert accuracy to negative log mean error
            #score = -math.log10(1-score)
            ax.axhline(score, linestyle='dotted', c=colours[model])
            ax.annotate(xy=(xpos, score-0.03), text=model.title(), c=colours[model])
