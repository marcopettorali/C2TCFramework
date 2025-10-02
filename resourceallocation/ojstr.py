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


class OJSTR:
    """
    OJSTR-like controller (simplified; delays provided; no storage; no device compute).
    All CPU quantities are in GHz; service load is in Gcycle; times are in ms.
    """

    def __init__(self, cloud_net_oneway_ms: float, cloud_cpu_capacity_ghz: float):
        self.cloud_net_oneway_ms = cloud_net_oneway_ms
        self.cloud = Cloud(cpu_capacity_ghz=cloud_cpu_capacity_ghz)
        self.services: Dict[int, Service] = {}
        self.devices: Dict[int, Device] = {}
        self.edges: Dict[int, EdgeNode] = {}

        # Book-keeping
        self._next_service_id = 0
        self._next_device_id = 0
        self._next_edge_id = 0

    # ------------- Dynamic registry -------------

    def add_edge(self, cpu_capacity_ghz: float) -> int:
        """
        Register an edge with capacity in **GHz** (e.g., 3.2).
        """
        eid = self._next_edge_id
        self._next_edge_id += 1
        self.edges[eid] = EdgeNode(id=eid, cpu_capacity_ghz=cpu_capacity_ghz)
        return eid

    def add_service(self, cycles_per_invocation_gcyc: float, deadline_ms: float) -> int:
        """
        Register a service: load in **Gcycle** (e.g., 0.75), deadline in **ms**.
        """
        sid = self._next_service_id
        self._next_service_id += 1
        self.services[sid] = Service(
            id=sid,
            cycles_per_invocation_gcyc=cycles_per_invocation_gcyc,
            deadline_ms=deadline_ms,
        )
        return sid

    def add_device(self, service_id: int, uplink_delay_to_edge_ms: Dict[int, float]) -> int:
        """
        Add a device bound to exactly ONE service with ONE persistent task.
        `uplink_delay_to_edge_ms` values are in **ms** (e.g., {edge0: 20.0}).
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
        return did

    # ===========================
    # Placement (ADD-ONLY, NO STORAGE)
    # ===========================

    def _service_placement_add_only(self) -> None:
        """
        For each edge, place any service not yet installed if there exists
        at least one device bound to that service such that the task would
        meet its deadline on this edge (all times in ms).

        Aggiunte: debug() con motivazioni chiare per 'placed' / 'not placed'.
        """
        debug("\n== SERVICE PLACEMENT (add-only) ==")
        for e in self.edges.values():
            debug(f"[edge {e.id}] capacity={e.cpu_capacity_ghz:.3f} GHz")
            for sid, s in self.services.items():
                if sid in e.placed_services:
                    # Evita rumore: commenta questa riga se preferisci non loggare i già piazzati
                    debug(f"  = service {sid}: already placed on edge {e.id}")
                    continue

                devices_for_service = [d for d in self.devices.values() if d.service_id == sid]
                if not devices_for_service:
                    debug(f"  ✗ service {sid}: no devices bound → not placed on edge {e.id}")
                    continue

                best = None  # (slack_ms, dev_id, delay_to_e_ms, edge_exec_ms, total_ms)
                no_link = 0
                for d in devices_for_service:
                    delay_to_e_ms = d.uplink_delay_to_edge_ms.get(e.id, math.inf)
                    if delay_to_e_ms == math.inf:
                        no_link += 1
                        continue
                    edge_exec_ms = 1000.0 * (s.cycles_per_invocation_gcyc / e.cpu_capacity_ghz)
                    total_ms = delay_to_e_ms + edge_exec_ms
                    slack_ms = s.deadline_ms - total_ms
                    if (best is None) or (slack_ms > best[0]):
                        best = (slack_ms, d.id, delay_to_e_ms, edge_exec_ms, total_ms)

                if best is None:
                    debug(f"  ✗ service {sid}: not placed on edge {e.id} — no reachable devices (missing uplink for {no_link} device(s))")
                    continue

                slack_ms, dev_id, delay_to_e_ms, edge_exec_ms, total_ms = best
                if slack_ms >= 0:
                    e.placed_services.add(sid)
                    debug(
                        "  ✓ place service {sid} on edge {eid}: "
                        "chosen dev {dev} → uplink={uplink:.1f} ms + exec={exec:.1f} ms "
                        "= {tot:.1f} ms ≤ deadline={dl:.1f} ms (slack {slack:.1f} ms)"
                        .format(sid=sid, eid=e.id, dev=dev_id,
                                uplink=delay_to_e_ms, exec=edge_exec_ms,
                                tot=total_ms, dl=s.deadline_ms, slack=slack_ms)
                    )
                else:
                    debug(
                        "  ✗ service {sid}: not placed on edge {eid} — best attempt with dev {dev}: "
                        "uplink={uplink:.1f} ms + exec={exec:.1f} ms = {tot:.1f} ms > deadline={dl:.1f} ms "
                        "(deficit {deficit:.1f} ms)"
                        .format(sid=sid, eid=e.id, dev=dev_id,
                                uplink=delay_to_e_ms, exec=edge_exec_ms,
                                tot=total_ms, dl=s.deadline_ms, deficit=-slack_ms)
                    )


    # ===========================
    # Public: Final allocation snapshot (persistent tasks)
    # ===========================

    def compute_final_allocation(self) -> Dict[str, Any]:
        """
        Final allocation WITH per-node CPU capacity (in GHz) and 1 persistent task per device.

        Steps (all time quantities in **ms**):
        1) Update placement (add-only).
        2) For each device, compute per-node CPU MIN required in **GHz** to meet deadline:
            f_req_ghz = 1000 * load_gcyc / slack_ms, where slack_ms = deadline_ms - net_delay_ms.
        3) Greedy "hardest-first": assign each device to the node that requires
            the SMALLEST f_req_ghz among nodes with enough residual capacity.
        4) On each node, allocate CPU = f_req_ghz + proportional share of residual,
            so the SUM of allocated GHz == node capacity (if node has ≥1 task).
        5) Compute final times with allocated GHz (compute_ms = 1000 * load_gcyc / GHz);
            build per-device and per-node views.

        Ritorna nel formato richiesto:
            {
            "BR0": [ { "process_name": "P3", "num_mns": 13, "cpu_share": 0.79 }, ... ],
            "BR1": [ ... ],
            ...
            "CN":  [ ... ]
            }
        """
        # 1) Placement
        self._service_placement_add_only()

        debug("\n== ALLOCATION (GHz/ms) ==")

        # 2) Build per-device candidates (ms) + debug
        devices = list(self.devices.values())
        dev_candidates: Dict[int, List[Dict[str, Any]]] = {}

        for d in devices:
            s = self.services[d.service_id]
            header = f"[dev {d.id} → service {s.id}] deadline={s.deadline_ms:.1f} ms"
            debug(header)
            cand = []

            # Edge options
            for e in self.edges.values():
                if d.service_id not in e.placed_services:
                    continue
                delay_up_ms = d.uplink_delay_to_edge_ms.get(e.id, math.inf)
                slack_ms = s.deadline_ms - delay_up_ms
                if math.isinf(delay_up_ms):
                    debug(f"  - edge {e.id}: no uplink → skip")
                    continue
                if slack_ms <= 0:
                    debug(f"  - edge {e.id}: uplink={delay_up_ms:.1f} ms → slack={slack_ms:.1f} ms ≤ 0 → infeasible")
                    continue
                f_req_ghz = 1000.0 * s.cycles_per_invocation_gcyc / slack_ms
                if f_req_ghz > 0:
                    cand.append({
                        "node_type": "edge",
                        "edge_id": e.id,
                        "f_req_ghz": f_req_ghz,
                        "comps": {
                            "uplink_ms": delay_up_ms,
                            "cloud_net_oneway_ms": None,
                            "slack_ms": slack_ms
                        }
                    })
                    debug(f"  + edge {e.id}: uplink={delay_up_ms:.1f} ms, slack={slack_ms:.1f} ms → f_req={f_req_ghz:.3f} GHz")

            # Cloud option (best uplink + one-way extra), all in ms
            if d.uplink_delay_to_edge_ms:
                best_uplink_ms = min(d.uplink_delay_to_edge_ms.values())
            else:
                best_uplink_ms = math.inf
            cloud_extra_ms = self.cloud_net_oneway_ms
            slack_ms = s.deadline_ms - (best_uplink_ms + cloud_extra_ms)

            if math.isinf(best_uplink_ms):
                debug("  - cloud: no uplink path (no edges reachable) → skip")
            elif slack_ms <= 0:
                debug(f"  - cloud: uplink={best_uplink_ms:.1f} ms + extra={cloud_extra_ms:.1f} ms "
                    f"→ slack={slack_ms:.1f} ms ≤ 0 → infeasible")
            else:
                f_req_ghz = 1000.0 * s.cycles_per_invocation_gcyc / slack_ms
                if f_req_ghz > 0:
                    cand.append({
                        "node_type": "cloud",
                        "edge_id": None,
                        "f_req_ghz": f_req_ghz,
                        "comps": {
                            "uplink_ms": best_uplink_ms,
                            "cloud_net_oneway_ms": cloud_extra_ms,
                            "slack_ms": slack_ms
                        }
                    })
                    debug(f"  + cloud: uplink={best_uplink_ms:.1f} ms + extra={cloud_extra_ms:.1f} ms, "
                        f"slack={slack_ms:.1f} ms → f_req={f_req_ghz:.3f} GHz")

            if not cand:
                debug("  ✗ no feasible node (no candidates)")

            dev_candidates[d.id] = cand

        # 3) Greedy assignment with capacity (hardest-first on min f_req_ghz)
        edge_cap: Dict[int, float] = {e.id: e.cpu_capacity_ghz for e in self.edges.values()}
        cloud_cap: float = self.cloud.cpu_capacity_ghz

        # Capacità iniziali
        cap_msg = [f"edge {eid}={cap:.3f} GHz" for eid, cap in edge_cap.items()]
        cap_msg.append(f"cloud={cloud_cap:.3f} GHz")
        debug("Initial capacities: " + ", ".join(cap_msg))

        def min_f_req(dev_id: int) -> float:
            c = dev_candidates[dev_id]
            if not c:
                return math.inf
            return min(opt["f_req_ghz"] for opt in c)

        dev_order = [d.id for d in self.devices.values()]
        debug("Assignment order (hardest-first by min f_req): " +
            ", ".join([f"dev {i} (min={min_f_req(i):.3f} GHz)" for i in dev_order]))

        assign_edge: Dict[int, List[Dict[str, Any]]] = {e.id: [] for e in self.edges.values()}
        assign_cloud: List[Dict[str, Any]] = []
        deferred: Dict[int, bool] = {d.id: False for d in devices}

        for dev_id in dev_order:
            cand = dev_candidates[dev_id]
            if not cand:
                deferred[dev_id] = True
                debug(f"  → dev {dev_id}: deferred (no candidates)")
                continue

            sorted_cand = sorted(cand, key=lambda x: x["f_req_ghz"])
            debug("  → dev {d}: trying nodes by f_req: {lst}".format(
                d=dev_id,
                lst=", ".join([
                    (f"edge {c['edge_id']}" if c["node_type"] == "edge" else "cloud") +
                    f" [{c['f_req_ghz']:.3f} GHz]"
                    for c in sorted_cand
                ])
            ))

            placed = False
            for c in sorted_cand:
                req = c["f_req_ghz"]
                if c["node_type"] == "edge":
                    eid = c["edge_id"]
                    if edge_cap[eid] >= req:
                        edge_cap[eid] -= req
                        assign_edge[eid].append({"dev_id": dev_id, "f_req_ghz": req, "comps": c["comps"]})
                        debug(f"    ✓ dev {dev_id} → EDGE {eid}: f_req={req:.3f} GHz "
                            f"(residual edge {eid}={edge_cap[eid]:.3f} GHz)")
                        placed = True
                        break
                    else:
                        debug(f"    · edge {eid} lacks capacity: need {req:.3f} GHz, have {edge_cap[eid]:.3f} GHz")
                else:  # cloud
                    if cloud_cap >= req:
                        cloud_cap -= req
                        assign_cloud.append({"dev_id": dev_id, "f_req_ghz": req, "comps": c["comps"]})
                        debug(f"    ✓ dev {dev_id} → CLOUD: f_req={req:.3f} GHz "
                            f"(residual cloud={cloud_cap:.3f} GHz)")
                        placed = True
                        break
                    else:
                        debug(f"    · cloud lacks capacity: need {req:.3f} GHz, have {cloud_cap:.3f} GHz")

            if not placed:
                deferred[dev_id] = True
                best_req = min(c["f_req_ghz"] for c in sorted_cand)
                max_edge = max(edge_cap.values()) if edge_cap else 0.0
                debug(f"  ✗ dev {dev_id}: deferred — min required {best_req:.3f} GHz, "
                    f"max residual edge={max_edge:.3f} GHz, cloud={cloud_cap:.3f} GHz")

        # 4) Distribute residual capacity to saturate nodes (con debug)
        def finalize_alloc(assigned: List[Dict[str, Any]], total_cap_ghz: float, node_label: str) -> List[Dict[str, Any]]:
            if not assigned:
                debug(f"[{node_label}] empty set → nothing to allocate")
                return []
            sum_req = sum(x["f_req_ghz"] for x in assigned)
            residual = max(total_cap_ghz - sum_req, 0.0)
            debug(f"[{node_label}] capacity={total_cap_ghz:.3f} GHz, sum_req={sum_req:.3f} GHz, residual={residual:.3f} GHz")
            if residual > 0 and sum_req > 0:
                for x in assigned:
                    x["f_alloc_ghz"] = x["f_req_ghz"] + residual * (x["f_req_ghz"] / sum_req)
            elif residual > 0 and sum_req == 0:
                extra = residual / len(assigned)
                for x in assigned:
                    x["f_alloc_ghz"] = extra
            else:
                for x in assigned:
                    x["f_alloc_ghz"] = x["f_req_ghz"]

            # guard: never exceed total capacity (within epsilon)
            denom = sum(a["f_alloc_ghz"] for a in assigned)
            if denom > 0 and denom > total_cap_ghz * 1.0000001:
                scale = total_cap_ghz / denom
                for x in assigned:
                    x["f_alloc_ghz"] *= scale
                denom = sum(a["f_alloc_ghz"] for a in assigned)

            # Dettaglio allocazioni
            for x in assigned:
                debug(f"  · {node_label}: dev {x['dev_id']} f_req={x['f_req_ghz']:.3f} → f_alloc={x['f_alloc_ghz']:.3f} GHz")
            debug(f"[{node_label}] final sum_alloc={denom:.3f} GHz\n")
            return assigned

        edge_final: Dict[int, List[Dict[str, Any]]] = {}
        for e in self.edges.values():
            edge_final[e.id] = finalize_alloc(assign_edge[e.id], e.cpu_capacity_ghz, f"edge {e.id}")

        cloud_final = finalize_alloc(assign_cloud, self.cloud.cpu_capacity_ghz, "cloud")

        # --------- COSTRUZIONE OUTPUT RICHIESTO (per nodo, aggregato per servizio) ---------
        result: Dict[str, List[Dict[str, Any]]] = {}

        # Edges: BR{i}
        for e in self.edges.values():
            groups: Dict[int, Dict[str, Any]] = {}  # sid -> {sum_alloc_ghz, devs:set}
            for x in edge_final[e.id]:
                dev_id = x["dev_id"]
                sid = self.devices[dev_id].service_id
                g = groups.setdefault(sid, {"sum_alloc_ghz": 0.0, "devs": set()})
                g["sum_alloc_ghz"] += x["f_alloc_ghz"]
                g["devs"].add(dev_id)

            items = []
            for sid, g in groups.items():
                frac = (g["sum_alloc_ghz"] / e.cpu_capacity_ghz) if e.cpu_capacity_ghz > 0 else 0.0
                items.append({
                    "process_name": f"P{sid}",
                    "num_mns": len(g["devs"]),
                    "cpu_share": round(frac, 2)
                })

            # ordinamento stabile: quota desc, poi nome processo
            items.sort(key=lambda it: (-it["cpu_share"], it["process_name"]))
            result[f"BR{e.id}"] = items

        # Cloud: CN
        groups_cloud: Dict[int, Dict[str, Any]] = {}
        for x in cloud_final:
            dev_id = x["dev_id"]
            sid = self.devices[dev_id].service_id
            g = groups_cloud.setdefault(sid, {"sum_alloc_ghz": 0.0, "devs": set()})
            g["sum_alloc_ghz"] += x["f_alloc_ghz"]
            g["devs"].add(dev_id)

        items_cn = []
        for sid, g in groups_cloud.items():
            frac = (g["sum_alloc_ghz"] / self.cloud.cpu_capacity_ghz) if self.cloud.cpu_capacity_ghz > 0 else 0.0
            items_cn.append({
                "process_name": f"P{sid}",
                "num_mns": len(g["devs"]),
                "cpu_share": round(frac, 2)
            })
        items_cn.sort(key=lambda it: (-it["cpu_share"], it["process_name"]))
        result["CN"] = items_cn

        # Log riassuntivo
        debug("\n== OUTPUT (per nodo, aggregato per servizio) ==")
        for node_key, arr in result.items():
            if not arr:
                debug(f"{node_key}: []")
            else:
                for rec in arr:
                    debug(f"{node_key}: {rec['process_name']}  num_mns={rec['num_mns']}  cpu_share={rec['cpu_share']:.2f}")

        return result



class OJSTRWrapper:
    def __init__(self, context: Context):
        self.context = context

        # --- Extract parameters from context
        # Extract cloud parameters
        cloud_net_oneway_ms = Link("CL", context.topology_graph["GW"]["CN"]["desc"]).delay_distribution.pdf.mean_value()
        debug(f"Using cloud one-way network delay = {cloud_net_oneway_ms} ms ")
        cloud_cpu_capacity_ghz = context.hosts["CN"].cpu_ghz
        debug(f"Using cloud CPU capacity = {cloud_cpu_capacity_ghz} GHz ")

        # Initialize OJSTR controller
        self.ojstr = OJSTR(cloud_net_oneway_ms, cloud_cpu_capacity_ghz)

        # Register edges
        self.host_to_id_map = {}
        for host_label, host in context.hosts.items():
            if host_label == "CN":
                continue
            edge_id = self.ojstr.add_edge(cpu_capacity_ghz=host.cpu_ghz)
            self.host_to_id_map[host_label] = edge_id
            debug(f"Registered edge {edge_id} with capacity {host.cpu_ghz} GHz")
        debug(f"Host to edge ID map: {self.host_to_id_map}")

        # register services
        self.service_to_id_map = {}
        for process_name, process in context.processes.items():
            deadline_ms = process.max_delay_ms

            # computing average # cycles per invocation (Gcycle)
            cycles_per_invocation_gcyc = (
                process.application.benchmark.distribution.pdf.mean_value()
                * process.application.benchmark.cpu_ghz
                / 1000  # / 1000 convert to ms
            )
            service_id = self.ojstr.add_service(cycles_per_invocation_gcyc=cycles_per_invocation_gcyc, deadline_ms=deadline_ms)
            self.service_to_id_map[process_name] = service_id
            debug(
                f"Added service {process_name} with {cycles_per_invocation_gcyc} average cycles per invocation and max latency {deadline_ms} ms"
            )

        # compute average communication links
        self.links = {}
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
    set_logging_level("info")
    context = JNecora.load_context_from_file("configs/scenario1_het1.json", cpu_shares_descriptor={"fair_shares": "num_processes"})
    ojstr = OJSTRWrapper(context)
     
    ojstr.add_1_mn("P0")

    plan = ojstr.compute_final_allocation()

    info(plan)
