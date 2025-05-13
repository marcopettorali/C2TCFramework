import numpy as np
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return [float(m - h), float(m + h)]


def avg(data):
    return sum(data) / len(data)


def ci_err(data):
    assert len(data) == 2
    return (data[1] - data[0]) / 2