import numpy as np

from networking.entities import Host, Process
from utils.distribution import Distribution
from utils.printing import print
from resourceallocation.jnecora import PACKET_LOSS_MS

_QUEUING_TIME_CACHE = {}


def _queuing_time_job(g_pmf, st_conv_at, n_mns, cache_index=None, i=0):

    global _QUEUING_TIME_CACHE
    g_pmf.label = f"g_pmf_{i}"

    # print(g_pmf, style="debug")

    # if n_mns == 1 immediately stop
    if i == n_mns - 1:
        return g_pmf

    # compute the cache key
    cache_key = f"{cache_index}_{i}"
    if cache_index is not None and cache_key in _QUEUING_TIME_CACHE:
        # immediately call the next iteration
        g_pmf = _QUEUING_TIME_CACHE[cache_key]
        return _queuing_time_job(g_pmf, st_conv_at, n_mns, cache_index, i + 1)

    # perform the convolution
    g_pmf: Distribution = g_pmf + st_conv_at

    # if the first percentile == PACKET_LOSS_MS, return a dirac delta at PACKET_LOSS_MS
    if g_pmf.percentile(1) >= PACKET_LOSS_MS:
        # print(f"Queuing time is out-of-scale for ({cache_index}, {n_mns}, {i}), returning a dirac delta at {PACKET_LOSS_MS}ms", style="warning")
        g_pmf = Distribution.dirac_delta(PACKET_LOSS_MS)
        _QUEUING_TIME_CACHE[cache_key] = g_pmf
        return g_pmf

    # resample positive part of the distribution to uniform grid from 0
    positive_xsys = [(x, y) for x, y in zip(*g_pmf.data) if x >= 0]

    # if the positive part is empty, return a dirac delta at 0
    if len(positive_xsys) == 0:
        g_pmf = Distribution.dirac_delta(0)
    else:
        # use linear interpolation to avoid negative values (monotonicity)
        f = lambda x: np.interp(x, *zip(*positive_xsys), left=0, right=0)
        new_xs = np.arange(0, positive_xsys[-1][0], Distribution.PRECISION)
        new_ys = f(new_xs)

        # add the negative mass (= 1 - integral positive) to the first value
        new_ys[0] += (1 - np.trapezoid(new_ys, new_xs)) / Distribution.PRECISION

        # create the new g_pmf
        g_pmf = Distribution(new_xs, new_ys).normalize()

    _QUEUING_TIME_CACHE[cache_key] = g_pmf

    return _queuing_time_job(g_pmf, st_conv_at, n_mns, cache_index, i + 1)


def _queuing_time_sddu_model(service_time_prob, n_mns, g, cache_index=None):

    service_time_prob.label = "service_time"

    spacing_ms = 15 if g != 1 else 30
    at_minus_pmf = Distribution.dirac_delta(-spacing_ms)

    # convolve the service time with the negative spacing
    st_conv_at = service_time_prob + at_minus_pmf

    # start with the first g_pmf
    g_pmf = Distribution.dirac_delta(0)

    ret = _queuing_time_job(g_pmf, st_conv_at, n_mns, cache_index)

    return ret


def qtime(gamma_exe, process: Process, host: Host, cache_index=None, g=None):
    # print(gamma_exe, process.mns, g, cache_index)
    return _queuing_time_sddu_model(gamma_exe, process.mns, g, cache_index)
