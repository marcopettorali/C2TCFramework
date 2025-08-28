from multiprocessing import Process

from networking.entities import Host
from resourceallocation.context import Context
from utils.distribution import Distribution


def compute_coverage_matrix(*args, **kwargs):

    context = kwargs["context"]
    number_of_aps = kwargs["number_of_aps"]

    matrix = {}
    for process_name in context["processes"]:
        matrix[process_name] = {}
        for ap in context["environment"].deployment:
            matrix[process_name][ap.label] = 1 / number_of_aps

    return matrix


def qtime(context: Context, gamma_exe: Distribution, process: Process, host: Host, cpu_share, cache_index=None, period_seconds=None):
    # ASSUMPTION: gamma_exe is constant, I take the median only
    gamma_exe_val = gamma_exe.percentile(50)

    # ASSUMPTION: all packets arrive at the same time
    # hence the queuing time is just the execution time * process.mns - 1 (the first packet does not wait)
    queuing_time = gamma_exe_val * (process.mns - 1)

    # If queuing_time is >= period, the system is not stable
    if period_seconds is not None and queuing_time >= period_seconds * 1000: # convert to milliseconds
        queuing_time = 10000

    return Distribution.dirac_delta(queuing_time)
