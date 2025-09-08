import numpy as np

from networking.entities import Host, Process
from resourceallocation.context import Context, load_context
from utils.distribution import Distribution
from utils.logging import debug, info
from resourceallocation.jnecora import PACKET_LOSS_MS

from time import time

_QUEUING_TIME_CACHE = {}


def _queuing_time_job(g_pmf, st_conv_at, n_mns, cpu_share, cache_prefix=None, i=0):
    global _QUEUING_TIME_CACHE
    g_pmf.label = f"g_pmf_{i}"

    # if n_mns == 1 immediately stop
    if i == n_mns - 1:
        return g_pmf

    # compute the cache key
    cache_key = f"{cache_prefix}_{i}"
    if cache_prefix is not None and cache_key in _QUEUING_TIME_CACHE:
        if cpu_share in _QUEUING_TIME_CACHE[cache_key]:
            # immediately call the next iteration
            g_pmf = _QUEUING_TIME_CACHE[cache_key][cpu_share]
            return _queuing_time_job(g_pmf, st_conv_at, n_mns, cpu_share, cache_prefix, i + 1)

        # else, if exists an entry with higher cpu share and has first percentile >= PACKET_LOSS_MS, return a dirac delta at PACKET_LOSS_MS
        higher_cpu_share = [k for k in _QUEUING_TIME_CACHE[cache_key] if k > cpu_share]
        if higher_cpu_share:
            for k in higher_cpu_share:
                if _QUEUING_TIME_CACHE[cache_key][k].percentile(1) >= PACKET_LOSS_MS:
                    return Distribution.dirac_delta(PACKET_LOSS_MS)

    # perform the convolution
    g_pmf: Distribution = g_pmf + st_conv_at

    # if the first percentile == PACKET_LOSS_MS, return a dirac delta at PACKET_LOSS_MS
    first_percentile = g_pmf.percentile(1)
    if first_percentile >= PACKET_LOSS_MS:
        g_pmf = Distribution.dirac_delta(PACKET_LOSS_MS)
        if not cache_key in _QUEUING_TIME_CACHE:
            _QUEUING_TIME_CACHE[cache_key] = {}
        _QUEUING_TIME_CACHE[cache_key][cpu_share] = g_pmf
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

    if cache_key not in _QUEUING_TIME_CACHE:
        _QUEUING_TIME_CACHE[cache_key] = {}
    _QUEUING_TIME_CACHE[cache_key][cpu_share] = g_pmf

    return _queuing_time_job(g_pmf, st_conv_at, n_mns, cpu_share, cache_prefix, i + 1)


_ST_CONV_AT_CACHE = {}


def _queuing_time_sddu_model(service_time_prob, n_mns, cpu_share, g, cache_prefix=None):
    global _ST_CONV_AT_CACHE

    service_time_prob.label = "service_time"

    spacing_ms = 15 if g != 1 else 30
    at_minus_pmf = Distribution.dirac_delta(-spacing_ms)

    # find st_conv_at in the cache
    cache_key = f"{cache_prefix}_{cpu_share}"
    if cache_key in _ST_CONV_AT_CACHE:
        st_conv_at = _ST_CONV_AT_CACHE[cache_key]
    else:
        # convolve the service time with the negative spacing
        st_conv_at = service_time_prob + at_minus_pmf
        _ST_CONV_AT_CACHE[cache_key] = st_conv_at

    # start with the first g_pmf
    g_pmf = Distribution.dirac_delta(0)

    ret = _queuing_time_job(g_pmf, st_conv_at, n_mns, cpu_share, cache_prefix)
 
    return ret


def qtime(context: Context, gamma_exe, process: Process, host: Host, cpu_share, cache_prefix=None, g=None):
    ret = _queuing_time_sddu_model(gamma_exe, process.mns, cpu_share, g, cache_prefix)
    return ret


if __name__ == "__main__":
    context = load_context(filename="configs/scenario1.json")

    process = context.processes["P0"]
    host = context.hosts["BR0"]
    cpu_share = 2.4 / host.cpu_ghz

    gamma_exe = (
        process.application.benchmark.distribution.pdf * ((process.application.benchmark.cpu_ghz / host.cpu_ghz) * (1 / cpu_share))
    ).normalize()

    for mns in range(1, 13 + 1):
        process.mns = mns
        gamma_que = qtime(context, gamma_exe, process, host, cpu_share, cache_prefix=f"{process.name}_{host.label}", g=4)
        debug(mns, gamma_que)
