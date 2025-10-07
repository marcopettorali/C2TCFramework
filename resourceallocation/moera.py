# moera.py

"""
MOERA (refactored) + Wrapper
============================

- MOERA: algoritmo core, indipendente da Context.
- MOERAWrapper: tutta la logica che dipende da Context (topologia, distribuzioni, ecc.).

Funzionamento (in breve)
------------------------
- Obiettivo (semplificato, coerente con il tuo codice): scegliere l'host che minimizza E_Q (ritardo medio di rete),
  tra quelli con capacità CPU residua sufficiente per soddisfare la domanda CPU del processo.
- Nessuna migrazione: una volta allocato, non si sposta nulla.
- 1 user = 1 MN; ogni chiamata a add_1_mn(process_name) aggiunge un MN del processo.

Parametri chiave
----------------
- merge_vms = False:
    * Domanda CPU per processo è una costante (GHz) calcolata una sola volta (nel wrapper).
    * Ogni arrivo consuma quella quantità fissa di GHz.
    * Su uno stesso host lo stesso processo genera più record (come nel tuo codice originale).

- merge_vms = True:
    * Per ogni processo e host c'è un profilo assoluto in GHz: profile[host][k] = GHz TOTALI per servire k MNs.
    * L'aggiunta del (k+1)-esimo MN su un host costa δ = profile[host][k+1] - profile[host][k] GHz di capacità residua.
    * Se il processo è già presente su un host, si **aggiorna** la stessa entry incrementando num_mns
      e portando allocated_cpu al valore **totale** perfilato (niente nuova entry).
"""

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field
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
    allocated_cpu_ghz: float  # totale per il processo su questo host
    cpu_share: float          # allocated_cpu_ghz / host_capacity


class MOERA:
    """
    Core dell'algoritmo, senza riferimenti al Context.
    Se merge_vms=False usa una domanda CPU fissa per processo.
    Se merge_vms=True usa profili assoluti (GHz) per host e #MN (k -> GHz totali).
    """

    def __init__(
        self,
        *,
        hosts_cpu_ghz: Dict[str, float],
        eq_delay_ms: Dict[Tuple[str, str], float],  # (process, host) -> ms
        merge_vms: bool = False,
        process_scalar_demand_ghz: Optional[Dict[str, float]] = None,
        process_profiles_ghz: Optional[Dict[str, Dict[str, Dict[int, Optional[float]]]]] = None,
    ):
        self.merge_vms = merge_vms

        # Capacità nodi
        self._host_capacity: Dict[str, float] = dict(hosts_cpu_ghz)
        self._residual: Dict[str, float] = dict(hosts_cpu_ghz)

        # Ritardi medi di rete (E_Q)
        self._eq_ms: Dict[Tuple[str, str], float] = dict(eq_delay_ms)

        # Richieste CPU (scalar) o profili
        self._scalar: Dict[str, float] = process_scalar_demand_ghz or {}
        self._profiles: Dict[str, Dict[str, Dict[int, Optional[float]]]] = process_profiles_ghz or {}

        # Mappa allocazioni: host -> List[_AllocRecord]
        self._allocation_map: Dict[str, List[_AllocRecord]] = {}

        # contatore k (MN) per (host, processo) quando merge_vms=True
        self._k_counts: Dict[Tuple[str, str], int] = {}

    # ------------- API -------------
    def add_1_mn(self, process_name: str) -> None:
        """
        Aggiunge 1 MN del processo. Sceglie l'host a minimo E_Q tra quelli con capacità residua sufficiente.
        - merge_vms=False: richiede self._scalar[process_name] GHz di capacità residua.
        - merge_vms=True: usa profilo assoluto e consuma δ = prof[host][k+1] - prof[host][k].
        """
        if not self.merge_vms and process_name not in self._scalar:
            raise ValueError(f"[MOERA] Missing scalar demand for process {process_name}")
        if self.merge_vms and process_name not in self._profiles:
            raise ValueError(f"[MOERA] Missing profile for process {process_name}")

        host_costs = {}  # host -> (cost, delta_needed_ghz, next_total_ghz or None)
        for host, capacity in self._host_capacity.items():
            # Quanto serve su questo host?
            if self.merge_vms:
                prof_h = self._profiles[process_name].get(host, None)
                if prof_h is None:
                    debug(f"  Host {host}: no profile for process {process_name} → skip")
                    continue
                k = self._k_counts.get((host, process_name), 0)
                curr_total = prof_h.get(k, 0.0)
                next_total = prof_h.get(k + 1, None)
                if next_total is None:
                    debug(f"  Host {host}: profile lacks k={k+1} for process {process_name} → skip")
                    continue
                delta_needed = next_total - curr_total
                if delta_needed <= 0:
                    debug(f"  Host {host}: non-positive delta={delta_needed:.3f} GHz for {process_name} → skip")
                    continue
                if self._residual[host] < delta_needed:
                    debug(f"  Host {host} cannot host {process_name}: need {delta_needed:.3f} GHz, have {self._residual[host]:.3f} GHz")
                    continue
                # costo = E_Q (ritardo medio)
                cost = self._eq_ms.get((process_name, host), float("inf"))
                host_costs[host] = (cost, delta_needed, next_total)
                debug(f"  Host {host} candidate: E_Q={cost:.2f} ms, δ={delta_needed:.3f} GHz (k→k+1)")

            else:
                need = self._scalar[process_name]
                if self._residual[host] < need:
                    debug(f"  Host {host} cannot host {process_name}: need {need:.3f} GHz, have {self._residual[host]:.3f} GHz")
                    continue
                cost = self._eq_ms.get((process_name, host), float("inf"))
                host_costs[host] = (cost, need, None)
                debug(f"  Host {host} candidate: E_Q={cost:.2f} ms, need={need:.3f} GHz")

        # Scelta host a minimo E_Q
        if not host_costs:
            info(f"No host can host process {process_name} (not enough CPU)")
            return

        selected_host = min(host_costs, key=lambda h: host_costs[h][0])
        cost, delta_needed, next_total = host_costs[selected_host]
        info(f"Selected host for process {process_name}: {selected_host} (E_Q={cost:.2f} ms)")

        # Applica decisione (consumo capacità e aggiornamento mappa)
        self._residual[selected_host] -= delta_needed

        # aggiornamento struttura per piano
        if self.merge_vms:
            # trova/crea record del processo su selected_host
            rec = None
            for r in self._allocation_map.get(selected_host, []):
                if r.process_name == process_name:
                    rec = r
                    break
            # aggiorna contatore k
            k_prev = self._k_counts.get((selected_host, process_name), 0)
            k_new = k_prev + 1
            self._k_counts[(selected_host, process_name)] = k_new

            if rec is None:
                self._allocation_map.setdefault(selected_host, []).append(
                    _AllocRecord(
                        process_name=process_name,
                        num_mns=1,
                        allocated_cpu_ghz=next_total,  # totale dal profilo
                        cpu_share=next_total / self._host_capacity[selected_host] if self._host_capacity[selected_host] > 0 else 0.0,
                    )
                )
            else:
                rec.num_mns = k_new
                rec.allocated_cpu_ghz = next_total  # totale dal profilo
                rec.cpu_share = (
                    next_total / self._host_capacity[selected_host] if self._host_capacity[selected_host] > 0 else 0.0
                )
            debug(
                f"  ✓ {process_name} on {selected_host}: +{delta_needed:.3f} GHz "
                f"(k={k_new}, total={next_total:.3f} GHz, residual={self._residual[selected_host]:.3f} GHz)"
            )

        else:
            # comportamento originale: entry separata per ogni MN
            self._allocation_map.setdefault(selected_host, []).append(
                _AllocRecord(
                    process_name=process_name,
                    num_mns=1,
                    allocated_cpu_ghz=delta_needed,  # qui la domanda è costante per MN
                    cpu_share=delta_needed / self._host_capacity[selected_host] if self._host_capacity[selected_host] > 0 else 0.0,
                )
            )
            debug(
                f"  ✓ {process_name} on {selected_host}: +{delta_needed:.3f} GHz "
                f"(residual={self._residual[selected_host]:.3f} GHz)"
            )

    def get_plan(self) -> Dict[str, List[Dict[str, float]]]:
        """
        Restituisce il piano corrente aggregato per host nel formato richiesto:
        {
          "BR0": [{"process_name":"P3","num_mns":13,"cpu_share":0.79}, ...],
          "CN":  [...]
        }
        - Se merge_vms=False e ci sono più record dello stesso processo sullo stesso host,
          qui li aggreghiamo (sommiamo num_mns e allocated_cpu).
        """
        result: Dict[str, List[Dict[str, float]]] = {}
        for host, recs in self._allocation_map.items():
            # aggrega per processo (serve anche per il caso non-merge)
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
            # ordina quota desc, poi nome
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
    def __init__(self, context: Context, *, merge_vms: bool):
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

        # 3) Domanda CPU per processo (scalar) OPPURE profili se merge_vms=True
        process_scalar_demand_ghz: Optional[Dict[str, float]] = None
        process_profiles_ghz: Optional[Dict[str, Dict[str, Dict[int, Optional[float]]]]] = None

        if not merge_vms:
            # Calcolo "una volta sola" delle domande CPU (come nel tuo codice, ma spostato qui).
            process_scalar_demand_ghz = {}
            for pname, process in self.context.processes.items():
                # E_Q medio (ms) su tutti gli host
                avg_network_delay = sum(
                    self.context.links["gamma_com"][(pname, hname)] for hname in self.context.hosts
                ) / max(1, len(self.context.hosts))

                # funzione che, dato cpu_ghz, restituisce il quantile alla min_reliability della
                # distribuzione dei tempi di esecuzione scalata (ms)
                # NB: usiamo le stesse chiamate a metodi del tuo codice.
                def gamma_func(cpu_ghz: float) -> float:
                    if cpu_ghz == 0:
                        return float("inf")
                    return (
                        process.application.benchmark.distribution.pdf
                        * (process.application.benchmark.cpu_ghz / cpu_ghz)
                    ).normalize().quantile(process.min_reliability)

                target_delay = process.max_delay_ms - avg_network_delay
                # Ricerca esponenziale + bisezione (stessa utility che avevi)
                min_cpu = exponential_search(gamma_func, target_delay, tolerance=0.1)
                process_scalar_demand_ghz[pname] = float(min_cpu)
                debug(f"[INIT] scalar CPU for {pname}: {min_cpu:.4f} GHz (target ms {target_delay:.1f})")

        else:
            # Profili assoluti per host e #MN: in GHz totali per k MN
            # Qui puoi usare lo stesso costruttore che usi per OJSTR,
            # ad esempio una utility interna (es. JNecoraUtilities.compute_min_cpu_ghz_ph)
            # che restituisca un dict {k -> GHz_totali}, oppure popolare da file.
            from resourceallocation.jnecora import JNecoraUtilities  # opzionale, se disponibile nel tuo progetto
            process_profiles_ghz = {}
            for pname in self.context.processes:
                prof_per_host: Dict[str, Dict[int, Optional[float]]] = {}
                for hname in self.context.hosts:
                    cpu_profile = JNecoraUtilities.compute_min_cpu_ghz_ph(self.context, pname, hname)
                    # cpu_profile atteso: {k -> GHz_totali} / None se non supportato
                    prof_per_host[hname] = dict(cpu_profile)
                process_profiles_ghz[pname] = prof_per_host
                debug(f"[INIT] profile (GHz totals) for {pname}: {prof_per_host}")

        # 4) Istanzia il core
        self.moera = MOERA(
            hosts_cpu_ghz=hosts_cpu_ghz,
            eq_delay_ms=eq_delay_ms,
            merge_vms=merge_vms,
            process_scalar_demand_ghz=process_scalar_demand_ghz,
            process_profiles_ghz=process_profiles_ghz,
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
        "configs/scenario1_het1.json",
        cpu_shares_descriptor={"cpu_share_precision": 0.01, "cpu_share_round_precision": 2}
    )

    # Caso A: comportamento originale (scalar)
    info("=== MOERAWrapper merge_vms=False ===")
    mw = MOERAWrapper(context, merge_vms=False)
    for _ in range(10):
        mw.add_1_mn("P0")
    
    mw.add_1_mn("P1")
    info(mw.get_plan())

    info("=== MOERAWrapper merge_vms=True ===")
    mw = MOERAWrapper(context, merge_vms=True)
    for _ in range(100):
        mw.add_1_mn("P0")
    
    mw.add_1_mn("P1")
    info(mw.get_plan())



    # Caso B: profili assoluti per host/#MN (come OJSTR merge_vms=True)
    # mw2 = MOERAWrapper(context, merge_vms=True)
    # mw2.add_1_mn("P0")
    # mw2.add_1_mn("P0")
    # mw2.add_1_mn("P1")
    # info(mw2.get_plan())
