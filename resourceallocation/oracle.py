from typing import Dict, Iterable, Tuple, Any
from ortools.sat.python import cp_model

_INFINITY = 1_000_000_000

def solve_mn_allocation(
    min_cpu_dict: Dict[Tuple[Any, Any, int], float],  # (p,h,m) -> cpu_percent_needed
    host_capacities_perc: Dict[Any, float],                            # h -> cpu_capacity_percent
    time_limit_seconds: float | None = None,
):
    """
    Maximize total MNs allocated subject to per-host CPU capacities.

    Decision:
      For each (p,h), choose exactly one m in available Ms (ideally 0..MAX_M).
      Binary var y[p,h,m] = 1 if level m is chosen for (p,h), else 0.

    Constraints:
      - Exactly one m per (p,h):  sum_m y[p,h,m] = 1
      - Host capacity:            sum_p sum_m cpu(p,h,m) * y[p,h,m] <= C[h]

    Objective:
      Maximize sum_p sum_h sum_m m * y[p,h,m]
    """
    processes, hosts, _ = map(set, zip(*min_cpu_dict.keys()))
    processes = sorted(list(processes))
    hosts = sorted(list(hosts))

    # Build available m values for each (p,h) from min_cpu_dict keys
    available_m: Dict[Tuple[Any, Any], list[int]] = {}
    for (p, h, m), c in min_cpu_dict.items():
        if (p, h) not in available_m:
            available_m[(p, h)] = []
        available_m[(p, h)].append(m)

    # Basic sanity: ensure every (p,h) has at least one option
    for p in processes:
        for h in hosts:
            if (p, h) not in available_m:
                raise ValueError(f"No gamma entries provided for pair (p={p}, h={h}). "
                                 f"Include at least m=0 with cpu=0 if unsupported.")

    # Model
    model = cp_model.CpModel()

    # Decision vars
    y = {}
    for p in processes:
        for h in hosts:
            for m in available_m[(p, h)]:
                if (p, h, m) not in min_cpu_dict:
                    continue
                y[(p, h, m)] = model.NewBoolVar(f"y_p{p}_h{h}_m{m}")

    # Exactly-one per (p,h)
    for p in processes:
        for h in hosts:
            vars_for_pair = [y[(p, h, m)] for m in available_m[(p, h)] if (p, h, m) in y]
            if not vars_for_pair:
                raise ValueError(f"No valid decision variables for (p={p}, h={h}).")
            model.Add(sum(vars_for_pair) == 1)

    # Host capacity constraints
    for h in hosts:
        if h not in host_capacities_perc:
            raise ValueError(f"Missing capacity C[{h}]")
        cpu_terms = []
        for p in processes:
            for m in available_m[(p, h)]:
                if (p, h, m) in y:
                    cpu_required = min_cpu_dict[(p, h, m)]
                    # OR-Tools CP-SAT uses integers; scale to avoid floating issues.
                    # Here we scale by 100 to support percentages with two decimals.
                    cpu_terms.append((int(round(cpu_required * 100)), y[(p, h, m)]))
        model.Add(
            sum(coeff * var for coeff, var in cpu_terms)
            <= int(round((host_capacities_perc[h] if host_capacities_perc[h] != float("inf") else _INFINITY) * 100))
        )

    # Objective: maximize total MNs allocated
    obj_terms = []
    for (p, h, m), var in y.items():
        obj_terms.append(m * var)
    model.Maximize(sum(obj_terms))

    # Optional time limit
    solver = cp_model.CpSolver()
    if time_limit_seconds is not None:
        solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    # A bit more search aggressiveness helps on larger instances
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    result = {
        "status": solver.StatusName(status),
        "objective_value": None,
        "selected_m_by_pair": {},   # (p,h) -> m*
        "host_cpu_usage": {},       # h -> used CPU (same unit as C[h])
        "host_cpu_slack": {},       # h -> slack CPU
        "total_MNs": None,
    }

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        chosen = {}
        total_MNs = 0
        host_used_scaled = {h: 0 for h in hosts}

        for p in processes:
            for h in hosts:
                chosen_m = None
                # Find the m with y=1
                for m in available_m[(p, h)]:
                    key = (p, h, m)
                    if key in y and solver.Value(y[key]) == 1:
                        chosen_m = m
                        total_MNs += m
                        host_used_scaled[h] += int(round(min_cpu_dict[(p, h, m)] * 100))
                        break
                chosen[(p, h)] = chosen_m

        result["selected_m_by_pair"] = chosen
        result["objective_value"] = total_MNs
        result["total_MNs"] = total_MNs
        for h in hosts:
            used = host_used_scaled[h] / 100.0
            result["host_cpu_usage"][h] = used
            result["host_cpu_slack"][h] = host_capacities_perc[h] - used

    return result


# --------------------------
# Minimal example (remove/replace with your data)
if __name__ == "__main__":
    P = ["P1", "P2"]
    hosts = ["H1", "H2"]
    MAX_M = 3

    # min_cpu_dict[(p,h,m)] = required CPU%
    min_cpu_dict = {
        ("P1", "H1", 0): 0.0, ("P1", "H1", 1): 10.0, ("P1", "H1", 2): 19.0, ("P1", "H1", 3): 30.0,
        ("P1", "H2", 0): 0.0, ("P1", "H2", 1): 12.0, ("P1", "H2", 2): 20.0, ("P1", "H2", 3): 34.0,
        ("P2", "H1", 0): 0.0, ("P2", "H1", 1): 11.0, ("P2", "H1", 2): 21.0, ("P2", "H1", 3): 28.0,
        ("P2", "H2", 0): 0.0, ("P2", "H2", 1): 9.0,  ("P2", "H2", 2): 18.0, ("P2", "H2", 3): 27.0,
    }

    # host capacities (% CPU available)
    C = {"H1": 40.0, "H2": 45.0}

    res = solve_mn_allocation(P, hosts, MAX_M, min_cpu_dict, C, time_limit_seconds=10)
    print("Status:", res["status"])
    print("Total MNs:", res["total_MNs"])
    print("Selected (p,h)->m:")
    for (p, h), m in res["selected_m_by_pair"].items():
        print(f"  ({p},{h}) -> m={m}")
    print("Host usage/slack:")
    for h in hosts:
        print(f"  {h}: used={res['host_cpu_usage'][h]}%, slack={res['host_cpu_slack'][h]}%")
