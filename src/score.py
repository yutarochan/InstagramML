"""
    Accuracy Thresholds (depends on account)
        instagood: 14.5%
        josecabaco: 18.5%
        kissinfashion: 9.5%
        beautifuldestinations; 10.5%
        etdieucrea: 14.5%
    You get one point if it is in within the range
"""

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer

def get_points(y, y_pred, threshold=0.10, **kwargs):
    # to prevent bug in math
    if isinstance(y, pd.Series):
        y = y.values
    elif not isinstance(y, np.ndarray):
        y = np.ndarray(y)
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    elif not isinstance(y_pred, np.ndarray):
        y_pred = np.ndarray(y_pred)
    # actual calculation
    error = np.abs((y - y_pred)/y)
    points = np.zeros_like(y, dtype=int)
    points[error < threshold] = 1
    return sum(points)

def instagram_scorer(threshold):
    """
    Use this function as a scorer to apply to the data. Returns number of points
    applied to the fit.
    :param threshold: Threshold for scoring
    :return: scoring function for use in sklearn. Model and evaluation using tools
    take a scoring parameter that controls what metric they apply to estimators evaluated.
    We want to maximize the number of points we get.
    """
    scorer = make_scorer(get_points, greater_is_better=True, threshold=threshold)
    return scorer


if __name__ == '__main__':
    y_pred = np.ones(100) + np.random.random(100) * 0.20
    y = np.ones(100)
    score = get_points(y, y_pred, threshold=0.10)
    scorer = instagram_scorer(0.1)
    print(score)
    print(scorer)
