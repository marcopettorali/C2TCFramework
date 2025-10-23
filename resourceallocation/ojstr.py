"""
OJSTR (simplified, deadline-driven) — Assumptions & Adaptation Notes
====================================================================

This implementation adapts "Online Joint Service placement, Task scheduling,
and Resource allocation" (OJSTR) to a minimal, cost-free / energy-free setting
focused on meeting per-task deadlines.

UNITS (this implementation)
---------------------------
- **All CPU rates** (edge, cloud, allocated shares) are in **GHz** (i.e., Gcycle/s).
- **All loads** (service compute per invocation) are in **Gcycle**.
- **All times** (deadlines, uplink, network, compute, totals) are in **milliseconds (ms)**.
  Time [ms] = 1000 * (Gcycle / GHz).

A) OJSTR Paper Assumptions (conceptual model)
---------------------------------------------
- Compute time on edge/cloud is cycles / cpu_rate.
- Uplink transmission depends on radio rates and task size in the original model.
- Optimization includes costs (energy, cloud tenancy) — not used here.
- Service placement may be reconfigured over time.

B) Our Adaptation Assumptions (differences vs paper)
----------------------------------------------------
- No monetary costs and no energy models.
- Devices have **no local compute**: tasks must run at edge or cloud.
- **One-to-one binding & persistent task**:
  each device is bound to **exactly one service** and has **exactly one task**
  that remains **persistently allocated** on a node (edge/cloud); when the
  device “recalls” the task, it reuses that allocation.
  -> API: add_device(service_id=..., uplink_delay_to_edge_ms=...).
- Cloud execution delay includes:
    (i) radio uplink delay (best device→edge uplink, **ms**),
    (ii) **one-way extra network delay** to reach the cloud: `cloud_net_oneway_ms`,
    (iii) cloud processing time **in ms** = 1000 * (Gcycle / cloud_GHz).
  The cloud does NOT consume storage/placement resources.
- Edge compute time IS modeled: edge compute **ms** = 1000 * (Gcycle / edge_GHz).
- Uplink is a **fixed delay** provided per (device, edge) pair:
  `uplink_delay_to_edge_ms[edge_id]` in **ms**. The **best** device→edge uplink
  (min ms) is reused as access path to the cloud (plus `cloud_net_oneway_ms`).
- **No storage modeling at edges** and **placement is add-only** (never remove/migrate).
- **Capacity enforcement (no slots)**: at allocation time we enforce
  ∑ allocated_GHz ≤ node_capacity_GHz and we **saturate** nodes distributing any residual.

C) Output
---------
`compute_final_allocation()` (non-mutating) returns:
- per device: chosen node (edge/cloud), timing breakdown **in ms**, **allocated CPU in GHz**,
  and **#devices** sharing that node;
- per node: aggregates (#tasks, #devices, **capacity in GHz**, **allocated sum in GHz**).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
import math

# (Facoltativi, lasciati per integrazione nel tuo progetto)
from networking.entities import Link
from resourceallocation.context import Context
from resourceallocation.jnecora import JNecora, JNecoraUtilities
from utils.distribution import Distribution
from utils.logging import info, set_logging_level, debug

# ===========================
# Data models (GHz / Gcycle / ms, no device compute)
# ===========================


@dataclass
class Service:
    id: int
    cycles_per_invocation_gcyc: float  # Gcycle per invocazione
    deadline_ms: float  # deadline in ms


@dataclass
class Device:
    id: int
    service_id: int
    uplink_delay_to_edge_ms: Dict[int, float] = field(default_factory=dict)  # ms


@dataclass
class EdgeNode:
    id: int
    cpu_capacity_ghz: float
    placed_services: set = field(default_factory=set)


@dataclass
class Cloud:
    cpu_capacity_ghz: float  # PROVIDED BY CALLER (main)


# ===========================
# Core controller (1 device ↔ 1 service, 1 persistent task per device)
# ===========================

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
import math

# (Facoltativi, lasciati per integrazione nel tuo progetto)
from networking.entities import Link  # noqa: F401
from resourceallocation.context import Context  # noqa: F401
from resourceallocation.jnecora import JNecora  # noqa: F401
from utils.distribution import Distribution  # noqa: F401
from utils.logging import info, set_logging_level, debug  # noqa: F401


# ===========================
# Data models (GHz / Gcycle / ms, no device compute)
# ===========================


@dataclass
class Service:
    id: int
    # Se merge_vms=False usiamo questo scalar:
    cycles_per_invocation_gcyc: float = 0.0
    # Se merge_vms=True usiamo questo profilo: host_key -> {num_mns -> cycles_gcyc}
    # host_key: int per edge_id, "CN" per cloud
    host_profile_gcyc: Optional[Dict[Union[int, str], Dict[int, float]]] = None
    deadline_ms: float = 0.0


@dataclass
class Device:
    id: int
    service_id: int
    uplink_delay_to_edge_ms: Dict[int, float] = field(default_factory=dict)  # ms


@dataclass
class EdgeNode:
    id: int
    cpu_capacity_ghz: float
    placed_services: set = field(default_factory=set)


@dataclass
class Cloud:
    cpu_capacity_ghz: float  # PROVIDED BY CALLER (main)


# ===========================
# Core controller (1 device ↔ 1 service, 1 persistent task per device)
# ===========================


class OJSTR:
    """
    OJSTR-like controller (simplified; delays provided; no storage; no device compute).
    All CPU quantities are in GHz; service load is in Gcycle; times are in ms.

    ONLINE / NO-REBALANCING:
    - Ogni volta che arriva un device, scegliamo e fissiamo il nodo (edge o cloud)
      allocando la quota minima richiesta per rispettare la deadline.
    - Non ribilanciamo mai le quote già assegnate.

    MERGE VMs:
    - merge_vms=False → allocazioni per-device (come prima).
    - merge_vms=True  → una sola istanza per (host, servizio) con contatore num_mns;
                        la CPU totale richiesta dipende da host e num_mns secondo un profilo
                        fornito dall’utente in add_service().
    """

    def __init__(self, cloud_net_oneway_ms: float, cloud_cpu_capacity_ghz: float, *, merge_vms: bool = False):
        self.cloud_net_oneway_ms = cloud_net_oneway_ms
        self.cloud = Cloud(cpu_capacity_ghz=cloud_cpu_capacity_ghz)
        self.merge_vms = merge_vms

        self.services: Dict[int, Service] = {}
        self.devices: Dict[int, Device] = {}
        self.edges: Dict[int, EdgeNode] = {}

        # Book-keeping ids
        self._next_service_id = 0
        self._next_device_id = 0
        self._next_edge_id = 0

        # Capacità residue
        self._edge_residual_ghz: Dict[int, float] = {}
        self._cloud_residual_ghz: float = cloud_cpu_capacity_ghz

        # Assegnazioni persistenti
        if not self.merge_vms:
            # per-device records
            self._assign_edge: Dict[int, List[Dict[str, Any]]] = {}  # per edge_id
            self._assign_cloud: List[Dict[str, Any]] = []
        else:
            # per-servizio aggregato
            # edge: eid -> sid -> {num_mns, f_alloc_ghz, slack_min_ms, dev_ids:set}
            self._assign_edge_agg: Dict[int, Dict[int, Dict[str, Any]]] = {}
            # cloud: sid -> {num_mns, f_alloc_ghz, slack_min_ms, dev_ids:set}
            self._assign_cloud_agg: Dict[int, Dict[str, Any]] = {}

        # Index per device (dove è stato messo)
        self._dev_assignment_index: Dict[int, Dict[str, Any]] = {}
        self._deferred_devices: set = set()

    # ------------- Helpers -------------

    def _cycles_profile_lookup(self, s: Service, node_type: str, node_key: Union[int, str], num_mns: int) -> Optional[float]:
        """Ritorna cycles_gcyc dal profilo se merge_vms=True; altrimenti None."""
        if not self.merge_vms:
            return None
        prof = s.host_profile_gcyc or {}
        key = "CN" if node_type == "cloud" else node_key
        table = prof.get(key)
        if not table:
            return None
        return table.get(num_mns)

    # ------------- Dynamic registry -------------

    def add_edge(self, cpu_capacity_ghz: float) -> int:
        """
        Register an edge with capacity in **GHz** (e.g., 3.2).
        Inizializza capacità residua e bucket allocazioni per l'edge.
        """
        eid = self._next_edge_id
        self._next_edge_id += 1
        self.edges[eid] = EdgeNode(id=eid, cpu_capacity_ghz=cpu_capacity_ghz)
        self._edge_residual_ghz[eid] = cpu_capacity_ghz
        if not self.merge_vms:
            self._assign_edge[eid] = []
        else:
            self._assign_edge_agg[eid] = {}
        debug(f"[INIT] edge {eid}: capacity={cpu_capacity_ghz:.3f} GHz (residual set)")
        return eid

    def add_service(self, cycles_per_invocation_gcyc: Union[float, Dict[Union[int, str], Dict[int, float]]], deadline_ms: float) -> int:
        """
        Register a service.

        merge_vms=False:
          - cycles_per_invocation_gcyc: float (Gcycle per invocazione).

        merge_vms=True:
          - cycles_per_invocation_gcyc: dict { host_key -> { num_mns -> cycles_gcyc } }
            host_key: int (edge_id) o "CN" per cloud.
        """
        sid = self._next_service_id
        self._next_service_id += 1

        if not self.merge_vms:
            if not isinstance(cycles_per_invocation_gcyc, (int, float)):
                raise ValueError("merge_vms=False: cycles_per_invocation_gcyc deve essere un float (Gcycle).")
            self.services[sid] = Service(
                id=sid,
                cycles_per_invocation_gcyc=float(cycles_per_invocation_gcyc),
                host_profile_gcyc=None,
                deadline_ms=deadline_ms,
            )
            debug(f"[INIT] service {sid}: load={float(cycles_per_invocation_gcyc):.6f} Gcyc, deadline={deadline_ms:.1f} ms")
        else:
            if not isinstance(cycles_per_invocation_gcyc, dict):
                raise ValueError("merge_vms=True: cycles_per_invocation_gcyc deve essere un dict {host->{n->cycles}}.")
            self.services[sid] = Service(
                id=sid,
                cycles_per_invocation_gcyc=0.0,
                host_profile_gcyc=cycles_per_invocation_gcyc,
                deadline_ms=deadline_ms,
            )
            debug(f"[INIT] service {sid}: PROFILE per-host loaded, deadline={deadline_ms:.1f} ms")
        return sid

    def add_device(self, service_id: int, uplink_delay_to_edge_ms: Dict[int, float]) -> int:
        """
        Add a device bound to exactly ONE service with ONE persistent task.
        Esegue anche l'AMMISSIONE ONLINE e l'ALLOCAZIONE immediata (no rebalancing).
        """
        if service_id not in self.services:
            raise ValueError(f"Unknown service_id {service_id}")

        did = self._next_device_id
        self._next_device_id += 1
        self.devices[did] = Device(
            id=did,
            service_id=service_id,
            uplink_delay_to_edge_ms=dict(uplink_delay_to_edge_ms),
        )
        debug(f"\n== ADMIT dev {did} (service {service_id}) ==")
        self._admit_and_allocate(did)  # decisione irrevocabile alla creazione
        return did

    # ===========================
    # ONLINE admission & allocation (no rebalancing)
    # ===========================

    def _admit_and_allocate(self, dev_id: int) -> None:
        d = self.devices[dev_id]
        s = self.services[d.service_id]
        debug(f"[dev {dev_id}] deadline={s.deadline_ms:.1f} ms  merge_vms={self.merge_vms}")

        if not self.merge_vms:
            # ---- Modalità classica (per-device) ----
            candidates: List[Dict[str, Any]] = []

            # Edge candidates
            for e in self.edges.values():
                delay_up_ms = d.uplink_delay_to_edge_ms.get(e.id, math.inf)
                if math.isinf(delay_up_ms):
                    debug(f"  - edge {e.id}: no uplink → skip")
                    continue
                slack_ms = s.deadline_ms - delay_up_ms
                if slack_ms <= 0:
                    debug(f"  - edge {e.id}: uplink={delay_up_ms:.1f} → slack={slack_ms:.1f} ≤ 0 → infeasible")
                    continue
                f_req_ghz = 1000.0 * s.cycles_per_invocation_gcyc / slack_ms
                candidates.append({"node_type": "edge", "edge_id": e.id, "f_req_ghz": f_req_ghz})
                debug(f"  + edge {e.id}: f_req={f_req_ghz:.3f} GHz (residual={self._edge_residual_ghz[e.id]:.3f})")

            # Cloud candidate
            best_uplink_ms = min(d.uplink_delay_to_edge_ms.values(), default=math.inf)
            if math.isinf(best_uplink_ms):
                debug("  - cloud: no uplink path → skip")
            else:
                net_ms = best_uplink_ms + self.cloud_net_oneway_ms
                slack_ms = s.deadline_ms - net_ms
                if slack_ms > 0:
                    f_req_ghz = 1000.0 * s.cycles_per_invocation_gcyc / slack_ms
                    candidates.append({"node_type": "cloud", "edge_id": None, "f_req_ghz": f_req_ghz})
                    debug(f"  + cloud: f_req={f_req_ghz:.3f} GHz (residual={self._cloud_residual_ghz:.3f})")
                else:
                    debug(f"  - cloud: slack={slack_ms:.1f} ≤ 0 → infeasible")

            # Capacity check and choose min f_req
            feasible = []
            for c in candidates:
                if c["node_type"] == "edge":
                    if self._edge_residual_ghz[c["edge_id"]] >= c["f_req_ghz"]:
                        feasible.append(c)
                    else:
                        debug(
                            f"    · edge {c['edge_id']} lacks capacity: need {c['f_req_ghz']:.3f}, "
                            f"have {self._edge_residual_ghz[c['edge_id']]:.3f}"
                        )
                else:
                    if self._cloud_residual_ghz >= c["f_req_ghz"]:
                        feasible.append(c)
                    else:
                        debug(f"    · cloud lacks capacity: need {c['f_req_ghz']:.3f}, " f"have {self._cloud_residual_ghz:.3f}")

            if not feasible:
                debug(f"  ✗ dev {dev_id}: DEFERRED — no feasible node with enough residual capacity")
                self._deferred_devices.add(dev_id)
                self._dev_assignment_index[dev_id] = {"state": "deferred"}
                return

            feasible.sort(key=lambda x: (x["f_req_ghz"], 0 if x["node_type"] == "edge" else 1))
            choice = feasible[0]
            f = choice["f_req_ghz"]

            if choice["node_type"] == "edge":
                eid = choice["edge_id"]
                # place on-demand
                if d.service_id not in self.edges[eid].placed_services:
                    self.edges[eid].placed_services.add(d.service_id)
                    debug(f"    → place service {d.service_id} on edge {eid}")
                self._edge_residual_ghz[eid] -= f
                self._assign_edge[eid].append({"dev_id": dev_id, "service_id": d.service_id, "f_alloc_ghz": f})
                self._dev_assignment_index[dev_id] = {"state": "edge", "edge_id": eid, "f_alloc_ghz": f}
                debug(f"    ✓ ASSIGN dev {dev_id} → EDGE {eid}: alloc={f:.3f} GHz " f"(residual {self._edge_residual_ghz[eid]:.3f})")
            else:
                self._cloud_residual_ghz -= 0  # f # ASSUMPTION: cloud has infinite parallelism
                self._assign_cloud.append({"dev_id": dev_id, "service_id": d.service_id, "f_alloc_ghz": f})
                self._dev_assignment_index[dev_id] = {"state": "cloud", "f_alloc_ghz": f}
                debug(f"    ✓ ASSIGN dev {dev_id} → CLOUD: alloc={f:.3f} GHz " f"(residual {self._cloud_residual_ghz:.3f})")
            return

        # ---- Modalità merge_vms=True ----
        candidates: List[Dict[str, Any]] = []

        # Edge candidates (incrementali)
        for e in self.edges.values():
            delay_up_ms = d.uplink_delay_to_edge_ms.get(e.id, math.inf)
            if math.isinf(delay_up_ms):
                debug(f"  - edge {e.id}: no uplink → skip")
                continue

            group = self._assign_edge_agg[e.id].get(d.service_id)
            prev_k = group["num_mns"] if group else 0

            # lookup profilo cycles per (eid, k+1)
            cycles_new = self._cycles_profile_lookup(s, "edge", e.id, prev_k + 1)
            if cycles_new is None:
                debug(f"  - edge {e.id}: profile lacks entry for num_mns={prev_k+1} → skip")
                continue

            # prev total
            if prev_k == 0:
                f_prev = 0.0
                slack_prev = float("inf")
            else:
                slack_prev = group["slack_min_ms"]
                cycles_prev = self._cycles_profile_lookup(s, "edge", e.id, prev_k)
                f_prev = 1000.0 * cycles_prev / slack_prev if slack_prev > 0 else math.inf

            # nuovo slack min
            slack_newdev = s.deadline_ms - delay_up_ms
            if slack_newdev <= 0:
                debug(f"  - edge {e.id}: uplink={delay_up_ms:.1f} → slack_new={slack_newdev:.1f} ≤ 0 → infeasible")
                continue
            slack_new = min(slack_prev, slack_newdev)

            f_total = 1000.0 * cycles_new / slack_new if slack_new > 0 else math.inf
            delta = max(0.0, f_total - f_prev)

            if delta == math.inf or f_total == math.inf:
                debug(f"  - edge {e.id}: resulting slack ≤0 → infeasible")
                continue

            candidates.append(
                {
                    "node_type": "edge",
                    "edge_id": e.id,
                    "delta_ghz": delta,
                    "f_total_ghz": f_total,
                    "new_slack_min_ms": slack_new,
                    "prev_k": prev_k,
                }
            )
            debug(
                f"  + edge {e.id}: k={prev_k}→{prev_k+1}, "
                f"δ={delta:.3f} GHz (residual={self._edge_residual_ghz[e.id]:.3f}), f_total={f_total:.3f}"
            )

        # Cloud candidate (incrementale)
        best_uplink_ms = min(d.uplink_delay_to_edge_ms.values(), default=math.inf)
        if math.isinf(best_uplink_ms):
            debug("  - cloud: no uplink path → skip")
        else:
            group_c = self._assign_cloud_agg.get(d.service_id)
            prev_k = group_c["num_mns"] if group_c else 0

            cycles_new = self._cycles_profile_lookup(s, "cloud", "CN", prev_k + 1)
            if cycles_new is None:
                debug(f"  - cloud: profile lacks entry for num_mns={prev_k+1} → skip")
            else:
                slack_prev = group_c["slack_min_ms"] if group_c else float("inf")
                cycles_prev = self._cycles_profile_lookup(s, "cloud", "CN", prev_k) if group_c else 0.0
                f_prev = 1000.0 * cycles_prev / slack_prev if group_c and slack_prev > 0 else 0.0

                net_ms = best_uplink_ms + self.cloud_net_oneway_ms
                slack_newdev = s.deadline_ms - net_ms
                if slack_newdev <= 0:
                    debug(f"  - cloud: net={net_ms:.1f} → slack_new={slack_newdev:.1f} ≤ 0 → infeasible")
                else:
                    slack_new = min(slack_prev, slack_newdev)
                    f_total = 1000.0 * cycles_new / slack_new if slack_new > 0 else math.inf
                    delta = max(0.0, f_total - f_prev)

                    if delta != math.inf and f_total != math.inf:
                        candidates.append(
                            {
                                "node_type": "cloud",
                                "edge_id": None,
                                "delta_ghz": delta,
                                "f_total_ghz": f_total,
                                "new_slack_min_ms": slack_new,
                                "prev_k": prev_k,
                            }
                        )
                        debug(
                            f"  + cloud: k={prev_k}→{prev_k+1}, "
                            f"δ={delta:.3f} GHz (residual={self._cloud_residual_ghz:.3f}), f_total={f_total:.3f}"
                        )
                    else:
                        debug("  - cloud: resulting slack ≤0 → infeasible")

        # Capacity check + scelta min δGHz (tie → edge)
        feasible = []
        for c in candidates:
            if c["node_type"] == "edge":
                if self._edge_residual_ghz[c["edge_id"]] >= c["delta_ghz"]:
                    feasible.append(c)
                else:
                    debug(
                        f"    · edge {c['edge_id']} lacks residual: need δ={c['delta_ghz']:.3f}, "
                        f"have {self._edge_residual_ghz[c['edge_id']]:.3f}"
                    )
            else:
                if self._cloud_residual_ghz >= c["delta_ghz"]:
                    feasible.append(c)
                else:
                    debug(f"    · cloud lacks residual: need δ={c['delta_ghz']:.3f}, " f"have {self._cloud_residual_ghz:.3f}")

        if not feasible:
            debug(f"  ✗ dev {dev_id}: DEFERRED — no feasible node with enough residual capacity (merge_vms)")
            self._deferred_devices.add(dev_id)
            self._dev_assignment_index[dev_id] = {"state": "deferred"}
            return

        feasible.sort(key=lambda x: (x["delta_ghz"], 0 if x["node_type"] == "edge" else 1))
        choice = feasible[0]

        if choice["node_type"] == "edge":
            eid = choice["edge_id"]
            delta = choice["delta_ghz"]
            f_total = choice["f_total_ghz"]
            new_slack_min = choice["new_slack_min_ms"]

            # place on-demand
            if d.service_id not in self.edges[eid].placed_services:
                self.edges[eid].placed_services.add(d.service_id)
                debug(f"    → place service {d.service_id} on edge {eid} (merge_vms)")

            # crea/aggiorna gruppo
            group = self._assign_edge_agg[eid].get(d.service_id)
            if not group:
                group = {"num_mns": 0, "f_alloc_ghz": 0.0, "slack_min_ms": float("inf"), "dev_ids": set()}
                self._assign_edge_agg[eid][d.service_id] = group

            group["num_mns"] += 1
            group["f_alloc_ghz"] = f_total
            group["slack_min_ms"] = new_slack_min
            group["dev_ids"].add(dev_id)

            # consumiamo solo il delta
            self._edge_residual_ghz[eid] -= delta
            self._dev_assignment_index[dev_id] = {"state": "edge", "edge_id": eid, "f_alloc_ghz": delta, "merged": True}
            debug(
                f"    ✓ MERGE dev {dev_id} into EDGE {eid}: +δ={delta:.3f} GHz "
                f"(f_total={f_total:.3f}; residual edge {eid}={self._edge_residual_ghz[eid]:.3f})"
            )
        else:
            delta = choice["delta_ghz"]
            f_total = choice["f_total_ghz"]
            new_slack_min = choice["new_slack_min_ms"]

            group = self._assign_cloud_agg.get(d.service_id)
            if not group:
                group = {"num_mns": 0, "f_alloc_ghz": 0.0, "slack_min_ms": float("inf"), "dev_ids": set()}
                self._assign_cloud_agg[d.service_id] = group

            group["num_mns"] += 1
            group["f_alloc_ghz"] = f_total
            group["slack_min_ms"] = new_slack_min
            group["dev_ids"].add(dev_id)

            self._cloud_residual_ghz -= delta
            self._dev_assignment_index[dev_id] = {"state": "cloud", "f_alloc_ghz": delta, "merged": True}
            debug(
                f"    ✓ MERGE dev {dev_id} into CLOUD: +δ={delta:.3f} GHz "
                f"(f_total={f_total:.3f}; residual cloud={self._cloud_residual_ghz:.3f})"
            )

    # ===========================
    # Public: Snapshot (NO mutations, NO rebalancing)
    # ===========================

    def compute_final_allocation(self) -> Dict[str, Any]:
        """
        Ritorna lo **stato corrente** aggregato per nodo/servizio nel formato richiesto:

            {
              "BR0": [ { "process_name": "P3", "num_mns": 13, "cpu_share": 0.79 }, ... ],
              "BR1": [ ... ],
              ...
              "CN":  [ ... ]
            }

        - Nessuna operazione di ricalcolo o ridistribuzione.
        - 'cpu_share' è la quota aggregata per servizio sul nodo, espressa come
          frazione della capacità del nodo (0..1), arrotondata a 2 decimali.
        """
        result: Dict[str, List[Dict[str, Any]]] = {}

        if not self.merge_vms:
            # Aggrega per servizio a partire dai record per-device (come prima)
            for e in self.edges.values():
                groups: Dict[int, Dict[str, Any]] = {}
                for rec in self._assign_edge[e.id]:
                    sid = rec["service_id"]
                    g = groups.setdefault(sid, {"sum_alloc_ghz": 0.0, "devs": set()})
                    g["sum_alloc_ghz"] += rec["f_alloc_ghz"]
                    g["devs"].add(rec["dev_id"])
                items = []
                for sid, g in groups.items():
                    frac = (g["sum_alloc_ghz"] / e.cpu_capacity_ghz) if e.cpu_capacity_ghz > 0 else 0.0
                    items.append({"process_name": f"P{sid}", "num_mns": len(g["devs"]), "cpu_share": round(frac, 2)})
                items.sort(key=lambda it: (-it["cpu_share"], it["process_name"]))
                result[f"BR{e.id}"] = items

            groups_c: Dict[int, Dict[str, Any]] = {}
            for rec in getattr(self, "_assign_cloud", []):
                sid = rec["service_id"]
                g = groups_c.setdefault(sid, {"sum_alloc_ghz": 0.0, "devs": set()})
                g["sum_alloc_ghz"] += rec["f_alloc_ghz"]
                g["devs"].add(rec["dev_id"])
            items_cn = []
            for sid, g in groups_c.items():
                frac = (g["sum_alloc_ghz"] / self.cloud.cpu_capacity_ghz) if self.cloud.cpu_capacity_ghz > 0 else 0.0
                items_cn.append({"process_name": f"P{sid}", "num_mns": len(g["devs"]), "cpu_share": round(frac, 2)})
            items_cn.sort(key=lambda it: (-it["cpu_share"], it["process_name"]))
            result["CN"] = items_cn
        else:
            # Usa direttamente le strutture aggregate
            for e in self.edges.values():
                items = []
                for sid, g in self._assign_edge_agg[e.id].items():
                    frac = (g["f_alloc_ghz"] / e.cpu_capacity_ghz) if e.cpu_capacity_ghz > 0 else 0.0
                    items.append({"process_name": f"P{sid}", "num_mns": g["num_mns"], "cpu_share": round(frac, 2)})
                items.sort(key=lambda it: (-it["cpu_share"], it["process_name"]))
                result[f"BR{e.id}"] = items

            items_cn = []
            for sid, g in self._assign_cloud_agg.items():
                frac = (g["f_alloc_ghz"] / self.cloud.cpu_capacity_ghz) if self.cloud.cpu_capacity_ghz > 0 else 0.0
                items_cn.append({"process_name": f"P{sid}", "num_mns": g["num_mns"], "cpu_share": round(frac, 2)})
            items_cn.sort(key=lambda it: (-it["cpu_share"], it["process_name"]))
            result["CN"] = items_cn

        # Log riassuntivo
        debug("\n== SNAPSHOT (per nodo, aggregato per servizio) ==")
        for node_key, arr in result.items():
            if not arr:
                debug(f"{node_key}: []")
            else:
                for rec in arr:
                    debug(f"{node_key}: {rec['process_name']}  num_mns={rec['num_mns']}  cpu_share={rec['cpu_share']:.2f}")

        return result


class OJSTRWrapper:
    def __init__(
        self,
        context: Context,
        *,
        merge_vms: bool = False,
    ):
        self.context = context
        self.merge_vms = merge_vms

        # --- Cloud params
        cloud_net_oneway_ms = Link("CL", context.topology_graph["GW"]["CN"]["desc"]).delay_distribution.pdf.mean_value()
        debug(f"Using cloud one-way network delay = {cloud_net_oneway_ms} ms ")
        cloud_cpu_capacity_ghz = context.hosts["CN"].cpu_ghz
        debug(f"Using cloud CPU capacity = {cloud_cpu_capacity_ghz} GHz ")

        # Initialize OJSTR controller
        self.ojstr = OJSTR(cloud_net_oneway_ms, cloud_cpu_capacity_ghz, merge_vms=merge_vms)

        # Register edges
        self.host_to_id_map: Dict[str, int] = {}
        for host_label, host in context.hosts.items():
            if host_label == "CN":
                continue
            edge_id = self.ojstr.add_edge(cpu_capacity_ghz=host.cpu_ghz)
            self.host_to_id_map[host_label] = edge_id
            debug(f"Registered edge {edge_id} with capacity {host.cpu_ghz} GHz")
        debug(f"Host to edge ID map: {self.host_to_id_map}")

        # Register services
        self.service_to_id_map: Dict[str, int] = {}

        for process_name, process in context.processes.items():
            deadline_ms = process.max_delay_ms

            # Stima scalar (fallback/degenere)
            benchmark_gcyc_avg = process.application.benchmark.cpu_ghz * (
                process.application.benchmark.distribution.pdf.mean_value() / 1000.0
            )  # conv ms->s
            debug(f"  - service {process_name}: benchmark mean → {benchmark_gcyc_avg:.6f} Gcyc")

            if not merge_vms:
                service_id = self.ojstr.add_service(cycles_per_invocation_gcyc=benchmark_gcyc_avg, deadline_ms=deadline_ms)
                debug(f"Added service {process_name} scalar={benchmark_gcyc_avg:.6f} Gcyc, deadline={deadline_ms} ms")
            else:
                # Compute gcyc for benchmark to have delay with reliability at the required percentile
                rel = process.min_reliability * 100
                benchmark_gcyc_prel = process.application.benchmark.cpu_ghz * (
                    process.application.benchmark.distribution.pdf.percentile(rel) / 1000.0
                )  # conv ms->s
                debug(f"  - service {process_name}: benchmark at reliability {rel:.3f} → {benchmark_gcyc_prel:.6f} Gcyc")

                # ASSUMPTION: linear scaling of cycles with CPU to get cycles_gcyc_avg from cycles_gcyc_prel
                scaling_factor = benchmark_gcyc_avg / benchmark_gcyc_prel
                debug(f"    · scaling factor to get avg from prel: {scaling_factor:.6f}")

                def _host_to_id(host_label: str) -> Union[int, str]:
                    return "CN" if host_label == "CN" else self.host_to_id_map[host_label]

                profile_gcyc = {}
                for host_label, host in context.hosts.items():
                    key = _host_to_id(host_label)
                    cpu_share_per_host = JNecoraUtilities.compute_min_cpu_share_ph(context, process_name, host_label)

                    # Compute gcyc with the CPU GHz required to have the delay at the reliability requested by the process
                    profile_gcyc.setdefault(key, {})
                    for num_mns, cpu_share_prel in cpu_share_per_host.items():
                        if cpu_share_prel is None:
                            profile_gcyc[key][num_mns] = None
                        else:
                            cpu_ghz_prel = cpu_share_prel * host.cpu_ghz
                            # cycles at the CPU GHz required to have the delay at the reliability requested by the process
                            gcyc_prel = cpu_ghz_prel * (
                                context.links["gamma_tot_precomputed"][(process.name, host.label, cpu_share_prel, num_mns)] / 1000.0
                            )  # conv ms->s
                            gcyc_avg = gcyc_prel * scaling_factor
                            profile_gcyc[key][num_mns] = gcyc_avg
                
                if host_label == "CN":  
                    debug(f"Service {process_name} profile for cloud: {profile_gcyc['CN']}", style="red bold")

                debug(f"Added service {process_name}, profile={profile_gcyc}, deadline={deadline_ms} ms")
                service_id = self.ojstr.add_service(cycles_per_invocation_gcyc=profile_gcyc, deadline_ms=deadline_ms)

            self.service_to_id_map[process_name] = service_id

        debug(f"Service to ID map: {self.service_to_id_map}")

        # compute average communication links (uplink ms)
        self.links: Dict[str, Dict[int, float]] = {}
        for process_name, process in context.processes.items():
            self.links[process_name] = {}
            for host_label, host in context.hosts.items():
                if host_label == "CN":
                    continue
                self.links[process_name][self.host_to_id_map[host_label]] = float(
                    self.context.links["gamma_com"][(process.name, host.label)].mean_value()
                )
        debug(f"Computed uplink delays (ms): {self.links}")

    def add_1_mn(self, process_name: str):
        self.ojstr.add_device(service_id=self.service_to_id_map[process_name], uplink_delay_to_edge_ms=self.links[process_name])

    def compute_final_allocation(self) -> Dict[str, Any]:
        return self.ojstr.compute_final_allocation()


# ===========================
# Minimal usage example (GHz/Gcycle; persistent 1 task per device)
# ===========================
if __name__ == "__main__":
    set_logging_level("debug")
    context = JNecora.load_context_from_file(
        "configs/scenario1_het1.json", cpu_shares_descriptor={"cpu_share_precision": 0.01, "cpu_share_round_precision": 2}
    )

    ojstr = OJSTRWrapper(context, merge_vms=True)

    for _ in range(8):
        ojstr.add_1_mn("P3")

    plan = ojstr.compute_final_allocation()

    info(plan)
