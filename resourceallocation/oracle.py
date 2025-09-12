from typing import Dict, Tuple, Any
from collections import defaultdict
from ortools.sat.python import cp_model
import itertools, argparse, sys, json, os
from argparse import RawTextHelpFormatter

_INFINITY = 1_000_000_000


def solve_mn_allocation(
    min_cpu_dict: Dict[Tuple[Any, Any, int], float],
    host_capacities_perc: Dict[Any, float],
    allocation_mode="max_mns",
    splitting_mode="optimal",
    time_limit_seconds: float | None = None,
):
    """Solve MN allocation with OR-Tools CP-SAT."""
    if allocation_mode not in ["max_mns", "max_apps_max_mns"]:
        raise ValueError(f"Unsupported allocation_mode {allocation_mode}")
    if splitting_mode not in ["optimal", "disabled"]:
        raise ValueError(f"Unsupported splitting_mode {splitting_mode}")

    processes, hosts, _ = map(set, zip(*min_cpu_dict.keys()))
    processes, hosts = sorted(processes), sorted(hosts)

    # Collect available m for each (p,h)
    available_m = defaultdict(list)
    for (p, h, m), _ in min_cpu_dict.items():
        available_m[(p, h)].append(m)

    model = cp_model.CpModel()
    # Decision vars: y[p,h,m] = 1 if we allocate m MNs of process p on host h
    y = {(p, h, m): model.NewBoolVar(f"y_{p}_{h}_{m}") for (p, h), ms in available_m.items() for m in ms}

    # Exactly one choice per (p,h)
    for (p, h), ms in available_m.items():
        model.Add(sum(y[(p, h, m)] for m in ms) == 1)

    # Host capacity constraints
    for h in hosts:
        terms = [int(round(min_cpu_dict[(p, h, m)] * 100)) * y[(p, h, m)] for p in processes for m in available_m[(p, h)]]
        cap = host_capacities_perc.get(h, 0.0)
        model.Add(sum(terms) <= int(round((cap if cap != float("inf") else _INFINITY) * 100)))

    # Splitting disabled: one host per process
    if splitting_mode == "disabled":
        for p in processes:
            uses = []
            for h in hosts:
                z = model.NewBoolVar(f"use_{p}_{h}")
                (
                    model.AddMaxEquality(z, [y[(p, h, m)] for m in available_m[(p, h)] if m > 0])
                    if any(m > 0 for m in available_m[(p, h)])
                    else model.Add(z == 0)
                )
                uses.append(z)
            model.Add(sum(uses) <= 1)

    # Objective
    total_mns = sum(m * var for (p, h, m), var in y.items())
    if allocation_mode == "max_mns":
        model.Maximize(total_mns)
    else:
        app_served = {p: model.NewBoolVar(f"app_{p}") for p in processes}
        for p in processes:
            (
                model.AddMaxEquality(app_served[p], [y[(p, h, m)] for h in hosts for m in available_m[(p, h)] if m > 0])
                if any(m > 0 for h in hosts for m in available_m[(p, h)])
                else model.Add(app_served[p] == 0)
            )
        BIG = 10**6
        model.Maximize(BIG * sum(app_served.values()) + total_mns)

    solver = cp_model.CpSolver()
    if time_limit_seconds:
        solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    # Build results
    res = {
        "status": solver.StatusName(status),
        "selected_m_by_pair": {},
        "host_cpu_usage": {},
        "host_cpu_slack": {},
        "total_MNs": 0,
        "total_apps_served": None if allocation_mode == "max_mns" else 0,
    }
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        host_used = {h: 0 for h in hosts}
        for p in processes:
            for h in hosts:
                for m in available_m[(p, h)]:
                    if solver.Value(y[(p, h, m)]) == 1:
                        res["selected_m_by_pair"][(p, h)] = m
                        res["total_MNs"] += m
                        host_used[h] += int(round(min_cpu_dict[(p, h, m)] * 100))
                        break
        for h in hosts:
            res["host_cpu_usage"][h] = host_used[h] / 100.0
            res["host_cpu_slack"][h] = host_capacities_perc[h] - res["host_cpu_usage"][h]
        if allocation_mode == "max_apps_max_mns":
            res["total_apps_served"] = sum(1 for p in processes if any(res["selected_m_by_pair"][(p, h)] > 0 for h in hosts))
    return res


if __name__ == "__main__":
    # CLI interface
    parser = argparse.ArgumentParser(
        description="J-NECORA: Resource allocation for C2TC (2024, Marco Pettorali)\nM. Pettorali, F. Righetti, C. Vallati, S. K. Das and G. Anastasi, \"J-NECORA: A Framework for Optimal Resource Allocation in Cloud-Edge-Things Continuum for Industrial Applications With Mobile Nodes,\" in IEEE Internet of Things Journal, vol. 12, no. 11, pp. 16525-16542, 1 June1, 2025, doi: 10.1109/JIOT.2025.3536700.\nhttps://ieeexplore.ieee.org/document/10886960",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("scenario_name", type=str, help="Scenario name relative to configs/")
    parser.add_argument(
        "--result-path", type=str, default="results.json", help="Path of the json file where to save results (relative to out/)"
    )
    parser.add_argument(
        "--allocation-mode", choices=["max_mns", "max_apps_max_mns"], default="max_mns", help="Allocation mode (default: max_mns)"
    )
    parser.add_argument(
        "--splitting-mode", choices=["optimal", "disabled"], default="optimal", help="Allow or forbid process splitting across hosts"
    )
    args = parser.parse_args()

    # Load context
    from resourceallocation.jnecora import JNecora
    from utils.logging import info  # adjust import path

    context = JNecora.load_context_from_file(f"configs/{args.scenario_name}.json")
    MAX_MNS = 13

    # Build min_cpu_dict
    min_cpu_dict = {(p, h, 0): 0.0 for p, h in itertools.product(context.processes, context.hosts)}
    for p, h in itertools.product(context.processes, context.hosts):
        max_delay_ms = context.processes[p].max_delay_ms
        tmp = defaultdict(list)
        for (pp, hh, cpu, mns), val in context.links["gamma_tot_precomputed"].items():
            if pp == p and hh == h and val <= max_delay_ms and mns <= MAX_MNS:
                tmp[mns].append(cpu)
        for k, vals in tmp.items():
            if vals:
                min_cpu_dict[(p, h, k)] = min(vals)

    host_caps = {h: float("inf") if host.infinite_parallelism else 1.0 for h, host in context.hosts.items()}
    ret = solve_mn_allocation(min_cpu_dict, host_caps, allocation_mode=args.allocation_mode, splitting_mode=args.splitting_mode)

    info(ret)
    split_by_host = {h: [] for h in context.hosts}
    for (p, h), m in ret["selected_m_by_pair"].items():
        if m > 0:
            split_by_host[h].append((p, m, float(min_cpu_dict[(p, h, m)])))
    info(split_by_host)

