# moera.py

"""
MOERA (refactored) + Wrapper — profili anche con merge_vms=False
================================================================

- MOERA: core senza dipendenze da Context.
- MOERAWrapper: tutto ciò che dipende da Context (topologia, distribuzioni, ecc.).

Comportamento
-------------
- Nessuna migrazione: una volta allocato, non si sposta.
- 1 user = 1 MN; ogni chiamata add_1_mn(process_name) aggiunge un MN.

Profili CPU (GHz)
-----------------
- `process_profiles_ghz[process][host][k] = GHz totali per servire k MN su *quel* host.
- merge_vms=True:
    * per aggiungere il (k+1)-esimo MN su (host, process) serve
      δ = profile[host][k+1] - profile[host][k] GHz di capacità residua.
    * si **aggiorna** la stessa voce in mappa (num_mns e allocated_cpu passano al totale `profile[host][k+1]`).

- merge_vms=False (nuovo comportamento richiesto):
    * si usa **sempre** l’entry con k=1 del profilo su ciascun host:
      `vm_need_ghz = profile[host][1]`.
    * Ogni MN ha la sua VM dedicata: ad ogni arrivo si consuma `vm_need_ghz`
      e si **aggiunge una nuova entry** (record separato) su quell’host.
      (Nel `get_plan()` vengono poi aggregati per host/processo.)
"""

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import itertools

from networking.entities import Link
from resourceallocation.context import Context
from resourceallocation.utils import find_paths
from utils.logging import debug, info


# ------------------------------------------------------------
# Helper: ritardi medi end-to-end tra processi e host (ms)
# ------------------------------------------------------------
def _compute_average_end_to_end_communication_delays(context: Context) -> Context:
    """
    Calcola i ritardi end-to-end medi (ms) tra ogni processo e host nella topologia.
    Salva in context.links["gamma_com"][(process, host)] = float(ms).
    """
    topology_graph = context.topology_graph
    context.links["gamma_com"] = {}

    processes = [n for n in topology_graph if topology_graph.nodes[n]["type"] == "process"]
    hosts = [n for n in topology_graph if topology_graph.nodes[n]["type"] == "host"]

    link_distributions = {}

    for process, host in itertools.product(processes, hosts):
        paths = find_paths(topology_graph, process, host)
        average_path_delay = 0.0
        for path in paths:
            path_delay = 0.0  # ms
            probability = 1.0
            for link in path:
                if "probability" in link["info"]:
                    probability *= link["info"]["probability"]
                key = (link["src"], link["dest"])
                if key not in link_distributions:
                    link_distributions[key] = Link(f"{link['src']}_{link['dest']}", delay_distribution=link["info"]["desc"])
                link_avg_delay = link_distributions[key].delay_distribution.pdf.mean_value()
                path_delay += link_avg_delay
            average_path_delay += path_delay * probability

        context.links["gamma_com"][(process, host)] = float(average_path_delay)
    return context


# ------------------------------------------------------------
# MOERA core (Context-free)
# ------------------------------------------------------------
@dataclass
class _AllocRecord:
    process_name: str
    num_mns: int
    allocated_cpu_ghz: float  # totale per questa *entry*
    cpu_share: float  # allocated_cpu_ghz / host_capacity


class MOERA:
    """
    Core dell'algoritmo, senza riferimenti al Context.

    - merge_vms=True  → usa profili assoluti per host e #MN, aggiorna un'unica voce per (host,process).
    - merge_vms=False → usa comunque i profili, ma *sempre* k=1 per ogni nuovo MN (VM dedicata).
    """

    def __init__(
        self,
        *,
        hosts_cpu_ghz: Dict[str, float],
        eq_delay_ms: Dict[Tuple[str, str], float],  # (process, host) -> ms
        process_profiles_ghz: Dict[str, Dict[str, Dict[int, Optional[float]]]],
        merge_vms: bool = False,
    ):
        self.merge_vms = merge_vms

        # Capacità nodi
        self._host_capacity: Dict[str, float] = dict(hosts_cpu_ghz)
        self._residual: Dict[str, float] = dict(hosts_cpu_ghz)

        # Ritardi medi di rete (E_Q)
        self._eq_ms: Dict[Tuple[str, str], float] = dict(eq_delay_ms)

        # Profili assoluti (GHz totali per k MN)
        # profiles[process][host][k] = GHz totali; None se non supportato
        self._profiles: Dict[str, Dict[str, Dict[int, Optional[float]]]] = process_profiles_ghz

        # Mappa allocazioni: host -> List[_AllocRecord]
        self._allocation_map: Dict[str, List[_AllocRecord]] = {}

        # contatore k (MN) per (host, processo) quando merge_vms=True
        self._k_counts: Dict[Tuple[str, str], int] = {}

    # ------------- API -------------
    def add_1_mn(self, process_name: str) -> None:
        """
        Aggiunge 1 MN del processo.
        - merge_vms=True: consumo marginale δ = prof[host][k+1] - prof[host][k].
        - merge_vms=False: consumo fisso vm_need = prof[host][1] (se esiste) — nuova entry per ogni MN.
        """
        if process_name not in self._profiles:
            raise ValueError(f"[MOERA] Missing profile for process {process_name}")

        host_costs = {}  # host -> (cost_ms, delta_needed_ghz, next_total_or_vm_need)
        for host, cap in self._host_capacity.items():
            prof_h = self._profiles[process_name].get(host, None)
            if prof_h is None:
                debug(f"  Host {host}: no profile for process {process_name} → skip")
                continue

            if self.merge_vms:
                # profilo cumulativo: totale per k MN
                k = self._k_counts.get((host, process_name), 0)
                curr_total = prof_h.get(k, 0.0)
                next_total = prof_h.get(k + 1, None)
                if next_total is None:
                    debug(f"  Host {host}: profile lacks k={k+1} for {process_name} → skip")
                    continue
                delta_needed = next_total - curr_total
                if delta_needed <= 0:
                    debug(f"  Host {host}: non-positive δ={delta_needed:.3f} GHz for {process_name} → skip")
                    continue
                if self._residual[host] < delta_needed:
                    debug(f"  Host {host}: need δ={delta_needed:.3f} GHz, have {self._residual[host]:.3f} GHz → skip")
                    continue
                cost = self._eq_ms.get((process_name, host), float("inf"))
                host_costs[host] = (cost, delta_needed, next_total)
                debug(f"  Host {host} candidate: E_Q={cost:.2f} ms, δ={delta_needed:.3f} GHz (k→k+1)")

            else:
                # VM dedicata: usa sempre k=1 (valore assoluto per *una* VM)
                vm_need = prof_h.get(1, None)
                if vm_need is None:
                    debug(f"  Host {host}: profile lacks k=1 for {process_name} → skip")
                    continue
                if vm_need <= 0:
                    debug(f"  Host {host}: non-positive vm_need={vm_need:.3f} GHz for {process_name} → skip")
                    continue
                if self._residual[host] < vm_need:
                    debug(f"  Host {host}: need {vm_need:.3f} GHz, have {self._residual[host]:.3f} GHz → skip")
                    continue
                cost = self._eq_ms.get((process_name, host), float("inf"))
                host_costs[host] = (cost, vm_need, vm_need)
                debug(f"  Host {host} candidate: E_Q={cost:.2f} ms, vm_need={vm_need:.3f} GHz (k=1)")

        # Scelta host a minimo E_Q
        if not host_costs:
            info(f"No host can host process {process_name} (not enough CPU)")
            return

        selected_host = min(host_costs, key=lambda h: host_costs[h][0])
        cost_ms, delta_needed, third = host_costs[selected_host]
        info(f"Selected host for process {process_name}: {selected_host} (E_Q={cost_ms:.2f} ms)")

        # Applica decisione
        self._residual[selected_host] -= delta_needed

        if self.merge_vms:
            # aggiorna voce unica per (host, process)
            rec = None
            for r in self._allocation_map.get(selected_host, []):
                if r.process_name == process_name:
                    rec = r
                    break
            k_prev = self._k_counts.get((selected_host, process_name), 0)
            k_new = k_prev + 1
            self._k_counts[(selected_host, process_name)] = k_new

            next_total = third  # GHz totali per k_new
            if rec is None:
                self._allocation_map.setdefault(selected_host, []).append(
                    _AllocRecord(
                        process_name=process_name,
                        num_mns=1,
                        allocated_cpu_ghz=next_total,
                        cpu_share=next_total / self._host_capacity[selected_host] if self._host_capacity[selected_host] > 0 else 0.0,
                    )
                )
            else:
                rec.num_mns = k_new
                rec.allocated_cpu_ghz = next_total
                rec.cpu_share = next_total / self._host_capacity[selected_host] if self._host_capacity[selected_host] > 0 else 0.0

            debug(
                f"  ✓ {process_name} on {selected_host}: +δ={delta_needed:.3f} GHz  "
                f"(k={k_new}, total={next_total:.3f} GHz, residual={self._residual[selected_host]:.3f} GHz)"
            )

        else:
            # nuova entry (VM dedicata per questo MN)
            vm_need = third  # = prof[host][1]
            self._allocation_map.setdefault(selected_host, []).append(
                _AllocRecord(
                    process_name=process_name,
                    num_mns=1,
                    allocated_cpu_ghz=vm_need,
                    cpu_share=vm_need / self._host_capacity[selected_host] if self._host_capacity[selected_host] > 0 else 0.0,
                )
            )
            debug(
                f"  ✓ {process_name} on {selected_host}: +{vm_need:.3f} GHz "
                f"(residual={self._residual[selected_host]:.3f} GHz) [dedicated VM, k=1 profile]"
            )

    def get_plan(self) -> Dict[str, List[Dict[str, float]]]:
        """
        Restituisce il piano corrente aggregato per host nel formato richiesto:
        {
          "BR0": [{"process_name":"P3","num_mns":13,"cpu_share":0.79}, ...],
          "CN":  [...]
        }
        - Con merge_vms=False, se ci sono più record dello stesso processo sullo stesso host,
          qui li aggreghiamo (sommiamo num_mns e allocated_cpu).
        """
        result: Dict[str, List[Dict[str, float]]] = {}
        for host, recs in self._allocation_map.items():
            # aggrega per processo
            agg: Dict[str, Dict[str, float]] = {}
            for r in recs:
                a = agg.setdefault(r.process_name, {"num_mns": 0, "allocated_cpu": 0.0})
                a["num_mns"] += r.num_mns
                a["allocated_cpu"] += r.allocated_cpu_ghz

            items = []
            cap = self._host_capacity[host]
            for pname, a in agg.items():
                cpu_share = a["allocated_cpu"] / cap if cap > 0 else 0.0
                items.append(
                    {
                        "process_name": pname,
                        "num_mns": int(a["num_mns"]),
                        "cpu_share": round(cpu_share, 2),
                    }
                )
            items.sort(key=lambda it: (-it["cpu_share"], it["process_name"]))
            result[host] = items

        # includi host senza allocazioni
        for host in self._host_capacity:
            result.setdefault(host, [])
        return result


# ------------------------------------------------------------
# Wrapper: tutto ciò che tocca Context
# ------------------------------------------------------------
class MOERAWrapper:
    def __init__(self, context: Context, *, merge_vms: bool = False):
        self.context = context
        self.merge_vms = merge_vms

        # 1) Ritardi medi su tutti i path processo→host
        _compute_average_end_to_end_communication_delays(self.context)

        # 2) Capacità host (GHz) e dizionario E_Q(process, host)
        hosts_cpu_ghz: Dict[str, float] = {h.label: h.cpu_ghz for h in self.context.hosts.values()}
        eq_delay_ms: Dict[Tuple[str, str], float] = {
            (pname, hname): float(self.context.links["gamma_com"][(pname, hname)])
            for pname in self.context.processes
            for hname in self.context.hosts
        }

        # 3) Profili assoluti (GHz totali per k MN) — usati sia con merge_vms=True sia con merge_vms=False (k=1)
        from resourceallocation.jnecora import JNecoraUtilities  # se presente nel tuo progetto

        process_profiles_ghz: Dict[str, Dict[str, Dict[int, Optional[float]]]] = {}
        for process_name, process in self.context.processes.items():
            prof_per_host: Dict[str, Dict[int, Optional[float]]] = {}
            for host_label, host in self.context.hosts.items():
                # This is the CPUs needed to get the required percentile for each number of MNs
                cpu_profile = JNecoraUtilities.compute_min_cpu_ghz_ph(self.context, process_name, host_label)
                # I need to convert them to the CPU needed to get average performance
                # ASSUMPTION: linear scaling of CPU vs response time
                # ASSUMPTION: linear scaling of response time vs number of MNs

                debug(cpu_profile)
                _old_cpu_profile = dict(cpu_profile)

                # let's compute the average time with the benchmark device for 1 MN
                benchmark_avg_time_1_mn = process.application.benchmark.distribution.pdf.mean_value()
                # let's compute the time at the required percentile
                benchmark_prel_time_1_mn = process.application.benchmark.distribution.pdf.percentile(process.min_reliability * 100)

                # compute the factor to scale the times
                if cpu_profile[1] is None or cpu_profile[1] == 0:
                    factor = 1.0
                else:
                    factor = benchmark_avg_time_1_mn / benchmark_prel_time_1_mn

                for num_mns, cpu in cpu_profile.items():
                    if cpu is not None:
                        cpu_profile[num_mns] = cpu * factor

                debug(cpu_profile)

                def check(x, y):
                    if x is not None and y is not None:
                        return x < y
                    else:
                        x = x if x is not None else float("inf")
                        y = y if y is not None else float("inf")
                        return x <= y

                assert all(
                    [check(x, y) for x, y in zip(cpu_profile.values(), _old_cpu_profile.values())]
                ), f"old profile: {_old_cpu_profile}, new profile: {cpu_profile}"

                prof_per_host[host_label] = dict(cpu_profile)
            process_profiles_ghz[process_name] = prof_per_host
            debug(f"[INIT] profile (GHz totals) for {process_name}: {prof_per_host}")

        # 4) Istanzia il core (usa SEMPRE i profili; con merge_vms=False userà k=1)
        self.moera = MOERA(
            hosts_cpu_ghz=hosts_cpu_ghz,
            eq_delay_ms=eq_delay_ms,
            process_profiles_ghz=process_profiles_ghz,
            merge_vms=merge_vms,
        )

    # ------------- API esterna -------------
    def add_1_mn(self, process_name: str) -> None:
        self.moera.add_1_mn(process_name)

    def get_plan(self) -> Dict[str, List[Dict[str, float]]]:
        return self.moera.get_plan()


# ------------------------------------------------------------
# Exponential search (presa dal tuo codice, invariata)
# ------------------------------------------------------------
def exponential_search(func, target_func_value, tolerance=0.001):
    old_x = 0.0
    x = 0.0001
    while func(x) > target_func_value:
        old_x = x
        x *= 2

    low = x
    high = old_x
    while low <= high:
        mid = (low + high) / 2
        mid_value = func(mid)
        if abs(mid_value - target_func_value) <= tolerance:
            return mid
        elif mid_value < target_func_value:
            high = mid
        else:
            low = mid
    return (low + high) / 2


if __name__ == "__main__":
    from resourceallocation.jnecora import JNecora
    from utils.logging import info, set_logging_level

    set_logging_level("info")

    context = JNecora.load_context_from_file(
        "configs/scenario1_het1.json", cpu_shares_descriptor={"cpu_share_precision": 0.01, "cpu_share_round_precision": 2}
    )

    # Caso A: comportamento originale (scalar)
    info("=== MOERAWrapper merge_vms=False ===")
    mw = MOERAWrapper(context, merge_vms=False)
    for _ in range(2):
        mw.add_1_mn("P0")

    mw.add_1_mn("P1")
    info(mw.get_plan())

    info("=== MOERAWrapper merge_vms=True ===")
    mw = MOERAWrapper(context, merge_vms=True)
    for _ in range(2):
        mw.add_1_mn("P0")

    mw.add_1_mn("P1")
    info(mw.get_plan())

    # Caso B: profili assoluti per host/#MN (come OJSTR merge_vms=True)
    # mw2 = MOERAWrapper(context, merge_vms=True)
    # mw2.add_1_mn("P0")
    # mw2.add_1_mn("P0")
    # mw2.add_1_mn("P1")
    # info(mw2.get_plan())
