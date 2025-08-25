import copy
import itertools
import json
import math
import multiprocessing as mp
import os
import pickle
import random
from datetime import datetime
from pathlib import Path

from resourceallocation.allocation_map import AllocationMap
from resourceallocation.jnecora_utils import compute_process_closestbr_map, compute_wireless_dist, queuing_time_sddu_model
from networking.entities import CloudNode, Host, Process, ProcessSplit
from resourceallocation.context import Context, load_config
from utils.distribution import Distribution
from utils.logging import print

os.system("export OMP_NUM_THREADS=1")


def find_min_positive_bisect(func, a, b, xtol=1e-1, ytol=1e-6, max_iter=100):
    """
    Finds the minimum x in [a, b] such that func(x) > 0, stopping when f(left) ≈ f(right).

    Parameters:
    - func: Function to evaluate.
    - a: Lower bound of search interval.
    - b: Upper bound of search interval.
    - xtol: Tolerance for stopping based on x-values.
    - ytol: Tolerance for stopping based on y-values (|f(left) - f(right)|).
    - max_iter: Maximum number of iterations.

    Returns:
    - The smallest x in [a, b] such that func(x) > 0.
    - Returns None if no such x exists.
    """
    f_a, f_b = func(a), func(b)

    if f_a > 0:
        return a  # Already positive at the left bound

    if f_b <= 0:
        return None  # No valid solution in range

    left, right = a, b
    f_left, f_right = f_a, f_b

    for _ in range(max_iter):
        mid = (left + right) / 2
        f_mid = func(mid)

        if f_mid > 0:
            right, f_right = mid, f_mid  # Move left
        else:
            left, f_left = mid, f_mid  # Move right

        # Stop if f(left) ≈ f(right)
        if abs(f_left - f_right) < ytol:
            break

        # Stop if x-values are close enough
        if abs(right - left) < xtol:
            break

    return right  # Minimum x where func(x) > 0


def _compute_gamma_com_for_each_process(context: Context):
    # Set the parameters
    context.other_params = {}
    context.other_params["wireless_delay_ms"] = 2.7
    context.other_params["packet_loss_delay_ms"] = 5000

    context.other_params["g"] = 4

    # initialize the links structures
    context.links["wireless"] = {}
    context.links["gamma_farbr"] = {}
    context.links["gamma_com"] = {}

    # for each process
    context.other_params["process_closestbr_map"] = {}
    for process_name in context.processes:
        process = context.processes[process_name]
        print(f"Computing wireless distribution for process {process.name}", style="info")
        # 1. compute the wireless distribution according to the packet loss
        context.links["wireless"][process.name] = compute_wireless_dist(process, context)

        print(
            f"Computing far BR communication distribution for process {process.name}",
            style="info",
        )
        # 2. compute the communication delay to get to the BR when the MNs are not directly connected
        context.links["gamma_farbr"][process.name] = context.links["wireless"][process.name].pdf + context.links["backbone"].pdf + context.links["backbone"].pdf

        print(f"Computing closest BR map for process {process.name}", style="info")
        # 3. for each br precompute the communication delay to get to the BR when the MNs are directly connected
        closest_br_map = compute_process_closestbr_map(context.environment, process)
        for br in context.environment.deployment:
            direct_communication_prob = closest_br_map.get(br.label, 0)

            print(
                f"Direct communication probability for process {process.name} with BR {br.label} = {direct_communication_prob}",
                direct_communication_prob,
                style="debug",
            )

            # combine the two communication delays
            context.links["gamma_com"][(process.name, br.label)] = Distribution.combine(
                [
                    context.links["wireless"][process.name].pdf,
                    context.links["gamma_farbr"][process.name],
                ],
                [direct_communication_prob, 1 - direct_communication_prob],
            )

        # add the gamma_com for the CN
        gamma_com_cl = context.links["wireless"][process.name].pdf + context.links["backbone"].pdf + context.links["cloud"].pdf

        context.links["gamma_com"][(process.name, "CN")] = gamma_com_cl

    return context


def compute_gamma_tot(context, process: Process, host: Host, cpu_share: float) -> Distribution:
    assert isinstance(process, Process), "process must be an instance of Process"
    assert isinstance(host, Host), "host must be an instance of Host"
    assert isinstance(cpu_share, (float, int)), f"cpu_share must be a float or a int, and is a {type(cpu_share)}"

    print(
        f"Computing delay at min reliability for process {process.name} on host {host.label} with CPU share = {cpu_share}",
        style="debug",
    )

    if cpu_share == 0:
        print(f"CPU share is 0, the process cannot run on the host", style="debug")
        return float("inf")

    # retrieve the communication delay to get to the BR when the MNs are directly connected
    gamma_com = context.links["gamma_com"][(process.name, host.label)]

    # print("Bisection job with scale", scale, style="debug")
    # compute the scaled execution delay (based on the host's CPU and the scale parameter)

    gamma_exe = (process.application.benchmark.distribution.pdf * (process.application.benchmark.cpu_ghz / host.cpu_ghz) * (1 / cpu_share)).normalize()

    if host.infinite_parallelism:
        print(
            f"Host {host.label} has infinite parallelism, the processing delay is equal to the execution delay",
            style="debug",
        )
        # the processing delay is equal to the execution delay
        gamma_proc = gamma_exe
    else:
        print(
            f"Computing the queuing time for process {process.name} on host {host.label}",
            style="debug",
        )
        # compute the queuing time based on the number of MNs and the execution delay
        gamma_que = queuing_time_sddu_model(
            gamma_exe,
            process.mns,
            context.other_params["g"],
            cache_index=f"{process.name}_{host.label}_{cpu_share}",
        ).normalize()

        # compute the processing delay (queuing + execution)
        gamma_proc = gamma_que + gamma_exe

    # compute the total delay (communication + processing)
    gamma_tot: Distribution = gamma_com + gamma_proc
    gamma_tot = gamma_tot.normalize()

    return gamma_tot


def compute_delay_at_min_reliability(context, process: Process, host: Host, cpu_share: float):
    """
    Computes the end-to-end delay at the min reliability percentile for a process on a host with a given CPU share.
    Returns the delay (in ms) at the min reliability percentile.
    Note that this method is agnostic to any existing allocation.
    """
    gamma_tot:Distribution = compute_gamma_tot(context, process, host, cpu_share)

    # compute the delay at the min reliability percentile
    delay_at_min_reliability = gamma_tot.quantile(process.min_reliability)

    # TODO remove 1.5 ms for numerical errors
    delay_at_min_reliability -= 1.5

    print(f"Delay at min reliability = {delay_at_min_reliability} ms", style="debug")

    # return the delay at the min reliability percentile
    return delay_at_min_reliability


def _job_precompute_total_delay(context, process_name, host_label):

    max_delay_ms = context.processes[process_name].max_delay_ms

    # fun = (
    #     lambda cpu_share: float(compute_delay_at_min_reliability(context, context.processes[process_name], context.hosts[host_label], cpu_share)) - max_delay_ms
    # )

    # return find_min_positive_bisect(fun, 0.01, 1)

    results = {}
    to_break = False
    for x in range(100, 0, -1):
        share = x / 100
        if not to_break:
            delay_at_rel = float(compute_delay_at_min_reliability(context, context.processes[process_name], context.hosts[host_label], share))
            print(f"Process {process_name} on host {host_label} with {share*100}% share: {delay_at_rel}")
            results[(process_name, host_label, share)] = delay_at_rel
            if delay_at_rel > max_delay_ms:
                to_break = True
                results[(process_name, host_label, share)] = float("inf")
        else:
            results[(process_name, host_label, share)] = float("inf")

    return results


def _precompute_total_delay(context):

    for process_name, mns in itertools.product(context.processes.keys(), range(1, 13 + 1)):
        temp = copy.deepcopy(context.processes[process_name])
        temp.mns = mns
        context.processes[f"{process_name}_mns{mns}"] = temp

    results = {}

    with mp.Pool(mp.cpu_count()) as pool:

        results = pool.starmap(
            _job_precompute_total_delay,
            ((context, process_name, host_label) for process_name, host_label in itertools.product(context.processes.keys(), context.hosts.keys())),
        )

    print(results)

    context.links["gamma_tot_precomputed"] = {}
    for res in results:
        for key in res.keys():
            context.links["gamma_tot_precomputed"][key] = res[key]

    return context


def _post_process_gamma_tot_precomputed(context):
    out = {}
    for key in context.links["gamma_tot_precomputed"].keys():
        _processname_mns, host_label, cpu_share = key

        if "mns" not in _processname_mns:
            continue

        process_name = _processname_mns.split("_mns")[0]
        mns = int(_processname_mns.split("_mns")[1])
        if (process_name, host_label) not in out:
            out[(process_name, host_label)] = {}

        delay_ms = context.links["gamma_tot_precomputed"][key]

        if mns not in out[(process_name, host_label)]:
            out[(process_name, host_label)][mns] = (cpu_share, delay_ms)

        if delay_ms != float("inf") and delay_ms >= out[(process_name, host_label)][mns][1]:
            out[(process_name, host_label)][mns] = (cpu_share, delay_ms)

    context.links["gamma_tot_precomputed"] = out

    return context


class DJNecora:
    def __init__(
        self,
        split_policy: str,
        selection_policy: str,
        cloud_policy: str = "prioritize_cn",
        multiple_process_splits_on_same_host_policy: str = "merge_splits",
    ):
        self.context = None

        self.splitting_policy: str = split_policy
        self.selection_policy: str = selection_policy
        self.cloud_policy: str = cloud_policy
        self.multiple_process_splits_on_same_host_policy: str = multiple_process_splits_on_same_host_policy

        self.allocation_map = AllocationMap()

    def set_context(self, context):
        assert isinstance(context, Context), "context must be an instance of Context"
        self.context = context
        print(f"Context set", style="debug")

    @staticmethod
    def load_context_from_file(config_path: str, pickle_context: bool = True):
        print(f"Importing context from file {config_path}", style="debug")

        # convert the config_path to a Path object
        config_path = Path(config_path)

        # check if in the parent folder there is a pickles folder and inside a pickle file with the same name as the config file
        pickle_path = config_path.parent / "pickles" / "djnecora" / f"{config_path.stem}.pkl"
        if pickle_path.exists():

            # unpickle the context and check if the extracted context.config_file_content is the same as the content of config file
            with open(pickle_path, "rb") as f:
                pickled_context = pickle.load(f)

            # load the config file
            with open(config_path, "r") as f:
                config = json.load(f)

            # check if the config file content is the same as the pickled context
            if pickled_context.config_file_content != config:
                raise Exception(f"The content of the pickled context { pickle_path} is different from the content of the config file {config_path}! Exiting")

            # print the date of last modification of the pickle file
            last_modified_time = datetime.fromtimestamp(pickle_path.stat().st_mtime)
            formatted_time = last_modified_time.strftime("%H:%M:%S, %A %d %B %Y")

            print(
                f"Found pickle file {pickle_path}, edited on {formatted_time}. Loading the context from the pickle file",
                style="warning",
            )

            # return the pickled context
            pickled_context.is_pickled = True
            return pickled_context

        # load the config file
        context = load_config(config_path)

        # compute the gamma_com for each process
        context = _compute_gamma_com_for_each_process(context)

        # precompute the total delay for each process
        context = _precompute_total_delay(context)

        # if pickle_context is True, pickle the context
        if pickle_context:
            # create the pickles folder if it does not exist
            pickles_folder = config_path.parent / "pickles" / "djnecora"
            pickles_folder.mkdir(exist_ok=True)

            # pickle the context
            with open(pickle_path, "wb") as f:
                pickle.dump(context, f)

            print(f"Pickled context to {pickle_path}", style="warning")

        return context

    def import_context_from_file(self, config_path: str, pickle_context: bool = True):
        self.set_context(DJNecora.load_context_from_file(config_path, pickle_context))

    # def get_delay_at_min_reliability_for_cpu_share(self, process: Process, host: Host, cpu_share: float):
    #     # approximate CPU share to lowest 0.01
    #     approximated_cpu_share = math.floor(cpu_share * 100) / 100

    #     if approximated_cpu_share == 0:
    #         print(f"CPU share is {cpu_share}, approximated to 0: the process cannot run on the host", style="warning")
    #         return float("inf")

    #     key = (f"{process.name}_mns{process.mns}", host.label, approximated_cpu_share)

    #     assert key in self.context.links["gamma_tot_precomputed"], f"Key {key} not found in precomputed values"

    #     return self.context.links["gamma_tot_precomputed"][key]

    def compute_min_cpu_share(self, process: Process, host: Host):
        """
        Compute the minimum CPU share that the process needs to run on the host exploiting the precomputed values in self.context.links["gamma_tot_precomputed"]
        Returns None if the process cannot run on the host with the available utilization
        """
        print(
            f"Computing min CPU share for process {process.name} on host {host.label}",
            style="info",
        )

        # get the available utilization of the host
        available_utilization_on_host = self.allocation_map.get_available_utilization(host.label)

        # check if the process has some splits and some of them are already allocated on the host
        process_splits_name_on_host = self.allocation_map.get_process_split_names_on_host_for_process(process.name, host.label)

        print(f"available utilization on host {host.label} = {available_utilization_on_host}", style="debug")

        # if there are splits and the policy is to merge them, compute the available utilization
        if process_splits_name_on_host != [] and self.multiple_process_splits_on_same_host_policy == "merge_splits":
            # if the policy was to merge the splits, there is at maximum one split on the host (otherwise it would have been merged)
            assert len(process_splits_name_on_host) == 1, "More than one split on the host"
            allocated_split_name = process_splits_name_on_host[0]
            allocated_split: Process = self.allocation_map.get_allocated_process_splits()[allocated_split_name]

            # emulate the merging of the current split and the one already allocated by removing the utilization of the allocated split
            allocated_split_utilization = self.allocation_map.get_host_and_cpushare_for_process(allocated_split_name)[1]

            if not host.infinite_parallelism:
                available_utilization_on_host += allocated_split_utilization

            print(
                f"A split of process {process.name} ({allocated_split_name}) is already allocated on host {host.label} with CPU share = {allocated_split_utilization}, the available utilization is {available_utilization_on_host}",
                style="debug_warning",
            )

            # use a copy of the original process to merge the MNs of the two splits
            process = copy.deepcopy(process)
            process.mns += allocated_split.mns

            print("The process after the merge is", process, style="debug_warning")

        elif process_splits_name_on_host != []:
            raise Exception(f"Unknown multiple process splits on same host policy '{self.multiple_process_splits_on_same_host_policy}'")

        # check if the process could run on this host with the available utilization
        if available_utilization_on_host == 0.0:
            print(
                f"Host {host.label} has no available utilization, the process cannot run on the host",
                style="error",
            )
            return None

        # retrieve the min cpu share from the precomputed values
        min_cpu_share, delay_ms = self.context.links["gamma_tot_precomputed"][(process.name, host.label)][process.mns]

        # check if the process can run on the host with the available utilization
        if delay_ms == float("inf") or min_cpu_share > available_utilization_on_host:
            print(f"Process {process.name} cannot run on host {host.label} with {process.mns} MNs", style="info")
            return None

        # if there are splits and the policy is to merge them, only return the delta CPU share
        if process_splits_name_on_host != [] and self.multiple_process_splits_on_same_host_policy == "merge_splits":
            assert (
                min_cpu_share >= allocated_split_utilization
            ), f"The min CPU share is less than the allocated split utilization ({min_cpu_share} < {allocated_split_utilization})"
            delta_cpu_share = min_cpu_share - allocated_split_utilization
            print(f"The extra CPU share needed to merge the splits is {delta_cpu_share}", style="debug_warning")
            return delta_cpu_share
        elif process_splits_name_on_host != []:
            raise Exception(f"Unknown multiple process splits on same host policy '{self.multiple_process_splits_on_same_host_policy}'")

        # return the min CPU share
        return min_cpu_share

    def get_max_mns_with_min_cpu_share(self, process: Process, host: Host):
        """
        Find the max number of MNs that can be allocated to the process exploiting the available CPU share of the host.
        Returns the max number of MNs and the minimum CPU share that can be allocated to the process.
        If no MNs can be allocated, returns (0, None).
        """

        temp_proc = copy.deepcopy(process)
        lo, hi = 1, process.mns

        print(f"Trying to allocate lo={lo} MNs", style="info")
        temp_proc.mns = lo
        last_lo_min_cpu_share = self.compute_min_cpu_share(temp_proc, host)
        if last_lo_min_cpu_share is None:
            print(
                f"Process {process.name} cannot run on host {host.label} with 1 MN",
                style="info",
            )
            return (0, None)

        print(f"Trying to allocate hi={hi} MNs", style="info")
        temp_proc.mns = hi
        min_cpu_share = self.compute_min_cpu_share(temp_proc, host)
        if min_cpu_share is not None:
            print(
                f"Process {process.name} can run on host {host.label} with {hi} MNs and CPU share={min_cpu_share}",
                style="info",
            )
            return (hi, min_cpu_share)

        # if got here, a solution exists
        # apply bisection until lo == hi - 1. At that point lo is the max number of MNs that can be allocated
        while lo != hi - 1:
            m = (lo + hi) // 2
            print(f"Trying to allocate m={m} MNs", style="info")
            temp_proc.mns = m
            min_cpu_share = self.compute_min_cpu_share(temp_proc, host)
            if min_cpu_share is None:
                # if m MNs were too much, try with less MNs
                hi = m
            else:
                # if m MNs can be allocated, try to allocate more MNs
                lo = m

                # if the process can run with the current CPU share, update the last_lo_min_cpu_share
                last_lo_min_cpu_share = min_cpu_share

        assert last_lo_min_cpu_share is not None, "last_lo_min_cpu_share MUST NOT be None!"
        print(
            f"Process {process.name} can run on host {host.label} with {lo} MNs and CPU share={last_lo_min_cpu_share}",
            style="info",
        )
        return (lo, last_lo_min_cpu_share)

    def execute_selection_policy(self, process: Process, candidates):
        print(f"Executing selection policy {self.selection_policy}", style="debug")

        # check if there are any candidates
        if candidates is None or len(candidates) == 0:
            print(f"No candidate hosts to select from for process {process.name}")
            return None

        # if there is only one candidate, return it
        if len(candidates) == 1:
            print(f"Only one candidate {candidates[0]} to select from for process {process.name}")
            return candidates[0]

        # if the CloudNode is a candidate, return it
        if self.cloud_policy == "prioritize_cn":
            assert len([c for c in candidates if isinstance(self.context.hosts[c["host_label"]], CloudNode)]) <= 1, "More than one CloudNode candidate"
            cloud_candidate = [c for c in candidates if isinstance(self.context.hosts[c["host_label"]], CloudNode)]
            if len(cloud_candidate) == 1:
                print(f"CloudNode candidate {cloud_candidate[0]} to select from for process {process.name}")
                return cloud_candidate[0]
        else:
            raise Exception(f"Unknown cloud policy {self.cloud_policy}")

        # if there are multiple candidates, apply the selection policy
        selection_policy_sanitized = self.selection_policy.lower().replace("-", "").replace(" ", "")

        if selection_policy_sanitized in ["firstfit", "ff"]:
            # return the first candidate (assuming that the list is sorted)
            return candidates[0]

        elif selection_policy_sanitized in ["nextfit", "nf"]:
            # return the second candidate (assuming that the list is sorted)
            return candidates[1]

        elif selection_policy_sanitized in ["bestfit", "bf"]:
            # among the hosts, select the one that has the minimum available CPU power
            return min(
                candidates,
                key=lambda x: self.allocation_map.get_available_utilization(x["host_label"]) * self.context.hosts[x["host_label"]].cpu_ghz,
            )

        elif selection_policy_sanitized in ["worstfit", "wf"]:
            # among the hosts, select the one that has the maximum available CPU power
            return max(
                candidates,
                key=lambda x: self.allocation_map.get_available_utilization(x["host_label"]) * self.context.hosts[x["host_label"]].cpu_ghz,
            )

        elif selection_policy_sanitized in ["random", "rand", "r"]:
            # return a random candidate
            return random.choice(candidates)

        else:
            raise Exception(f"Unknown selection policy {self.selection_policy}")

    def apply_splitting_and_allocate(self, process: Process):
        """
        Try to allocate the process on the hosts according to the splitting policy.
        Returns the number of MNs left to allocate, 0 if all the MNs have been allocated.
        """

        print(f"Applying splitting policy {self.splitting_policy} for process {process.name}", style="info")

        # sanitize the split policy
        splitting_policy_sanitized = self.splitting_policy.lower().replace("-", "").replace(" ", "")

        # iteratively try to allocate as many MNs as possible
        mns_to_allocate = int(process.mns)
        while mns_to_allocate > 0:
            # build a temporary process to try to allocate the MNs
            temp_process = copy.deepcopy(process)

            # address the splitting policy
            if splitting_policy_sanitized in ["ramconstrained", "rc"] or splitting_policy_sanitized in ["nosplits"]:
                # try to allocate the biggest split possible
                temp_process.mns = mns_to_allocate
            elif splitting_policy_sanitized in ["ramunconstrained", "ru"]:
                # try to allocate the smallest split possible (=1 MN)
                temp_process.mns = 1
            else:
                raise Exception(f"Unknown split policy {self.splitting_policy}")

            # iterate the hosts and find the candidates that can accomodate the MNs
            candidates = []
            for host_label in self.context.hosts:
                host: Host = self.context.hosts[host_label]

                # if there are splits on the host to be merged, do NOT count the available RAM as a constraint
                if (
                    self.multiple_process_splits_on_same_host_policy == "merge_splits"
                    and len(self.allocation_map.get_process_split_names_on_host_for_process(process.name, host_label)) > 0
                ):
                    # if the policy is to merge the splits, do not consider the host if there are already splits of the same process allocated
                    print(
                        f"Host {host_label} already has a split of process {process.name} allocated and the policy is to merge the splits. RAM constraint is not considered",
                        style="info",
                    )
                else:
                    # retrieve the available RAM of the host and compare it with the RAM needed by the process
                    available_ram_gb = self.allocation_map.get_available_ram_gb(host_label)

                    # check if the host has enough RAM to allocate the process
                    if temp_process.application.ram_occupancy_gb > available_ram_gb:
                        print(
                            f"Host {host_label} does not have enough RAM to allocate process {temp_process.name} (requested RAM = {temp_process.application.ram_occupancy_gb} GB, available RAM = {available_ram_gb} GB)",
                            style="info",
                        )
                        continue

                # if the RAM constraint has been satisfied, check the CPU constraint

                # compute max number of MNs and min CPU share that can be allocated to the process on the host
                supported_mns, min_cpu_share = self.get_max_mns_with_min_cpu_share(temp_process, host)

                # only append the host if it can allocate at least 1 MN
                if supported_mns > 0:
                    candidates.append({"host_label": host_label, "supported_mns": supported_mns, "min_cpu_share": min_cpu_share})

            if splitting_policy_sanitized in ["nosplits"]:
                # if the policy is to not split the process, only support candidates that can accomodate all the MNs
                candidates = [c for c in candidates if c["supported_mns"] == process.mns]
            if splitting_policy_sanitized in ["ramconstrained", "rc"]:
                # Do NOT consider all candidates!
                # With the RAM-constrained policy we want to maximize the split size!
                # Hence, only consider the hosts that can allocate the maximum number of supported MNs
                candidates = [c for c in candidates if c["supported_mns"] == max([x["supported_mns"] for x in candidates])]

            # select the candidates that can accomodate all the MNs to allocate
            selected_candidate = self.execute_selection_policy(process, candidates)

            # check if at least one host can accomodate the MNs, otherwise exit the loop
            if selected_candidate == None:
                print(
                    f"No host can accomodate the available {mns_to_allocate} MNs of process {process.name}!",
                    style="error",
                )
                return mns_to_allocate

            # else allocate the process with some MNs on the selected host
            # unpack the selected candidate
            host_label, allocated_mns, min_cpu_share = (
                selected_candidate["host_label"],
                selected_candidate["supported_mns"],
                selected_candidate["min_cpu_share"],
            )
            print(
                f"Selection policy chose host {host_label} to allocate {allocated_mns} MNs of process {process.name}",
                style="info",
            )

            # create a new process with the allocated MNs to be allocated on the host
            temp_process.mns = allocated_mns

            # check if it is necessary to merge the splits
            if self.multiple_process_splits_on_same_host_policy == "merge_splits":
                merge_splits = True
            else:
                merge_splits = False

            # allocate the process on the host
            temp_process.__class__ = ProcessSplit
            temp_process: ProcessSplit
            temp_process.parent_process_name = process.name
            splits = self.allocation_map.get_splits_name_for_process(process.name)
            num_splits = 0 if splits is None else len(splits)
            temp_process.split_name = f"{process.name}_{num_splits}"  # (MNs={allocated_mns}/{process.mns})"
            new_process_added = self.allocation_map.allocate(temp_process, host_label, min_cpu_share, merge_splits)

            # if this allocation caused a new split to be added
            # register that the newly created process is a split of the original process
            if new_process_added:
                self.allocation_map.add_process_split_labels_map(process.name, temp_process.name)

            # update the number of MNs left to allocate
            mns_to_allocate -= allocated_mns

        # if got here, all the MNs have been allocated
        assert mns_to_allocate == 0, "All the MNs should have been allocated at this point!"

        # return number of MNs left = 0
        return 0

    def deploy(self, process: Process):

        if process is None:
            print("No process to deploy", style="warning")
            return

        assert isinstance(process, Process), "process must be an instance of Process"

        # check if the process is already deployed
        if self.allocation_map.get_host_label_for_process(process.name) is not None:
            raise Exception(f"Process {process.name} is already deployed")

        print(f"Deploying process {process.name}", style="info")

        # try to allocate the process on the hosts based on the splitting policy
        mns_left = self.apply_splitting_and_allocate(process)

        # check if there are MNs left
        if mns_left == 0:
            print(f"Process {process.name} deployed successfully", style="success")

        else:
            print(
                f"Process {process.name} could not be fully deployed, {mns_left} MNs left out of {process.mns}!",
                style="error",
            )

        return mns_left


if __name__ == "__main__":
    import itertools

    from networking.channel_model import override_channel_model_path

    override_channel_model_path("mobile6tisch-3.json")

    scenario_name = "scenario1_het1"
    context = DJNecora.load_context_from_file(f"configs/{scenario_name}.json")
    context = _post_process_gamma_tot_precomputed(context)

    # print(context.links["gamma_tot_precomputed"])
    # exit()

    context.other_params["g"] = 4
    context.other_params["min_cpu_share"] = 0.05

    selection_policies = ["first-fit", "next-fit", "best-fit", "worst-fit", "random"]
    split_policies = ["ram-constrained", "ram-unconstrained", "no-splits"]

    context.processes = {k: v for k, v in context.processes.items() if "mns" not in k}

    num_repetitions = 50

    results_to_dump = {}
    for rep_index in range(num_repetitions):

        # scramble the order of the processes
        process_names = list(context.processes.keys())
        random.shuffle(process_names)

        results_to_dump[rep_index] = {}
        results_to_dump[rep_index]["processes"] = process_names

        print(f"Repetition {rep_index}, processes: {process_names}")

        # iterate through the selection policies
        for selection_policy, split_policy in itertools.product(selection_policies, split_policies):
            print(f"Selection policy: {selection_policy}, Split policy: {split_policy}")

            # create the DJNecora object
            djnecora = DJNecora(selection_policy=selection_policy, split_policy=split_policy)
            djnecora.set_context(context)
            djnecora.allocation_map.set_hosts(context.hosts)

            # deploy all the processes
            for process_name in process_names:
                process = context.processes[process_name]
                djnecora.deploy(process)

            # print the allocation map and the allocated processes
            print(djnecora.allocation_map.to_rich_text())
            print(djnecora.allocation_map.get_allocated_process_splits())

            # "collect the collectable"
            allocated_processes = djnecora.allocation_map.get_allocated_process_splits()
            allocated_processes_mns = [{p_name: allocated_processes[p_name].mns} for p_name in allocated_processes]
            allocation_map = djnecora.allocation_map._allocation_map

            # store the results
            results_to_dump[rep_index][f"{selection_policy}, {split_policy}"] = {
                "allocated_processes_mns": allocated_processes_mns,
                "allocation_map": allocation_map,
            }

            # dump the results iteratively
            with open(f"results_{scenario_name}.json", "w") as f:
                json.dump(results_to_dump, f, indent=2)