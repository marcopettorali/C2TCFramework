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
        """
        for e in self.edges.values():
            for sid, s in self.services.items():
                if sid in e.placed_services:
                    continue
                feasible = False
                for d in self.devices.values():
                    if d.service_id != sid:
                        continue
                    delay_to_e_ms = d.uplink_delay_to_edge_ms.get(e.id, math.inf)
                    if delay_to_e_ms == math.inf:
                        continue
                    edge_exec_ms = 1000.0 * (s.cycles_per_invocation_gcyc / e.cpu_capacity_ghz)
                    if delay_to_e_ms + edge_exec_ms <= s.deadline_ms:
                        feasible = True
                        break
                if feasible:
                    e.placed_services.add(sid)

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
        """
        # 1) Placement
        self._service_placement_add_only()

        # 2) Build per-device candidates (ms)
        devices = list(self.devices.values())
        dev_candidates: Dict[int, List[Dict[str, Any]]] = {}
        for d in devices:
            s = self.services[d.service_id]
            cand = []

            # Edge options
            for e in self.edges.values():
                if d.service_id not in e.placed_services:
                    continue
                delay_up_ms = d.uplink_delay_to_edge_ms.get(e.id, math.inf)
                slack_ms = s.deadline_ms - delay_up_ms
                if slack_ms <= 0:
                    continue
                # f_req_ghz = cycles / (slack_s) = 1000 * cycles / slack_ms
                f_req_ghz = 1000.0 * s.cycles_per_invocation_gcyc / slack_ms
                if f_req_ghz > 0:
                    cand.append(
                        {
                            "node_type": "edge",
                            "edge_id": e.id,
                            "f_req_ghz": f_req_ghz,
                            "comps": {"uplink_ms": delay_up_ms, "cloud_net_oneway_ms": None},
                        }
                    )

            # Cloud option (best uplink + one-way extra + cloud compute), all in ms
            best_uplink_ms = min(d.uplink_delay_to_edge_ms.values(), default=math.inf)
            cloud_extra_ms = self.cloud_net_oneway_ms
            slack_ms = s.deadline_ms - (best_uplink_ms + cloud_extra_ms)
            if slack_ms > 0:
                f_req_ghz = 1000.0 * s.cycles_per_invocation_gcyc / slack_ms
                if f_req_ghz > 0:
                    cand.append(
                        {
                            "node_type": "cloud",
                            "edge_id": None,
                            "f_req_ghz": f_req_ghz,
                            "comps": {"uplink_ms": best_uplink_ms, "cloud_net_oneway_ms": cloud_extra_ms},
                        }
                    )

            dev_candidates[d.id] = cand

        # 3) Greedy assignment with capacity (hardest-first on min f_req_ghz)
        edge_cap: Dict[int, float] = {e.id: e.cpu_capacity_ghz for e in self.edges.values()}
        cloud_cap: float = self.cloud.cpu_capacity_ghz

        def min_f_req(dev_id: int) -> float:
            c = dev_candidates[dev_id]
            if not c:
                return math.inf
            return min(opt["f_req_ghz"] for opt in c)

        dev_order = sorted([d.id for d in devices], key=min_f_req, reverse=True)

        assign_edge: Dict[int, List[Dict[str, Any]]] = {e.id: [] for e in self.edges.values()}
        assign_cloud: List[Dict[str, Any]] = []
        deferred: Dict[int, bool] = {d.id: False for d in devices}

        for dev_id in dev_order:
            cand = dev_candidates[dev_id]
            if not cand:
                deferred[dev_id] = True
                continue

            placed = False
            for c in sorted(cand, key=lambda x: x["f_req_ghz"]):
                if c["node_type"] == "edge":
                    eid = c["edge_id"]
                    if edge_cap[eid] >= c["f_req_ghz"]:
                        edge_cap[eid] -= c["f_req_ghz"]
                        assign_edge[eid].append({"dev_id": dev_id, "f_req_ghz": c["f_req_ghz"], "comps": c["comps"]})
                        placed = True
                        break
                else:  # cloud
                    if cloud_cap >= c["f_req_ghz"]:
                        cloud_cap -= c["f_req_ghz"]
                        assign_cloud.append({"dev_id": dev_id, "f_req_ghz": c["f_req_ghz"], "comps": c["comps"]})
                        placed = True
                        break
            if not placed:
                deferred[dev_id] = True

        # 4) Distribute residual capacity to saturate nodes
        def finalize_alloc(assigned: List[Dict[str, Any]], total_cap_ghz: float) -> List[Dict[str, Any]]:
            if not assigned:
                return []
            sum_req = sum(x["f_req_ghz"] for x in assigned)
            residual = max(total_cap_ghz - sum_req, 0.0)
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
            return assigned

        edge_final: Dict[int, List[Dict[str, Any]]] = {}
        for e in self.edges.values():
            edge_final[e.id] = finalize_alloc(assign_edge[e.id], e.cpu_capacity_ghz)

        cloud_final = finalize_alloc(assign_cloud, self.cloud.cpu_capacity_ghz)

        # 5) Build per-device view and node stats (times in ms)
        devices_on_edge = {e.id: set() for e in self.edges.values()}
        for eid, lst in edge_final.items():
            for x in lst:
                devices_on_edge[eid].add(x["dev_id"])
        devices_on_cloud = set(x["dev_id"] for x in cloud_final)

        by_device: Dict[int, Dict[str, Any]] = {
            d.id: {"service_id": d.service_id, "assigned": None, "deferred": deferred[d.id]} for d in self.devices.values()
        }

        # Edge entries
        for eid, lst in edge_final.items():
            e = self.edges[eid]
            for x in lst:
                d = self.devices[x["dev_id"]]
                s = self.services[d.service_id]
                uplink_ms = x["comps"]["uplink_ms"]
                edge_compute_ms = 1000.0 * s.cycles_per_invocation_gcyc / x["f_alloc_ghz"] if x["f_alloc_ghz"] > 0 else math.inf
                total_ms = uplink_ms + edge_compute_ms
                by_device[x["dev_id"]]["assigned"] = {
                    "where": "edge",
                    "edge_id": eid,
                    "total_time_ms": total_ms,
                    "components": {
                        "uplink_ms": uplink_ms,
                        "edge_compute_ms": edge_compute_ms,
                        "cloud_net_oneway_ms": None,
                        "cloud_compute_ms": None,
                    },
                    "cpu_share_ghz": x["f_alloc_ghz"],
                    "num_devices_on_chosen_node": len(devices_on_edge[eid]),
                }
                by_device[x["dev_id"]]["deferred"] = False

        # Cloud entries
        for x in cloud_final:
            d = self.devices[x["dev_id"]]
            s = self.services[d.service_id]
            uplink_ms = x["comps"]["uplink_ms"]
            cloud_oneway_ms = x["comps"]["cloud_net_oneway_ms"]
            cloud_compute_ms = 1000.0 * s.cycles_per_invocation_gcyc / x["f_alloc_ghz"] if x["f_alloc_ghz"] > 0 else math.inf
            total_ms = uplink_ms + cloud_oneway_ms + cloud_compute_ms
            by_device[x["dev_id"]]["assigned"] = {
                "where": "cloud",
                "edge_id": None,
                "total_time_ms": total_ms,
                "components": {
                    "uplink_ms": uplink_ms,
                    "edge_compute_ms": None,
                    "cloud_net_oneway_ms": cloud_oneway_ms,
                    "cloud_compute_ms": cloud_compute_ms,
                },
                "cpu_share_ghz": x["f_alloc_ghz"],
                "num_devices_on_chosen_node": len(devices_on_cloud),
            }
            by_device[x["dev_id"]]["deferred"] = False

        node_stats = {
            "edges": {
                e.id: {
                    "num_tasks": len(edge_final[e.id]),
                    "num_devices": len(devices_on_edge[e.id]),
                    "cpu_capacity_ghz": e.cpu_capacity_ghz,
                    "cpu_allocated_ghz": sum(it["f_alloc_ghz"] for it in edge_final[e.id]) if edge_final[e.id] else 0.0,
                }
                for e in self.edges.values()
            },
            "cloud": {
                "num_tasks": len(cloud_final),
                "num_devices": len(devices_on_cloud),
                "cpu_capacity_ghz": self.cloud.cpu_capacity_ghz,
                "cpu_allocated_ghz": sum(it["f_alloc_ghz"] for it in cloud_final) if cloud_final else 0.0,
            },
        }

        return {"by_device": by_device, "node_stats": node_stats}


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
            cycles_per_invocation_gcyc = process.application.benchmark.distribution.pdf.mean_value() * process.application.benchmark.cpu_ghz
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
    set_logging_level("DEBUG")
    context = JNecora.load_context_from_file("configs/scenario1_het1.json", cpu_shares_descriptor={"fair_shares": "num_processes"})
    ojstr = OJSTRWrapper(context)
    ojstr.add_1_mn("P0")

    plan = ojstr.compute_final_allocation()

    info(plan)
    exit()

    # # Cloud: 500 GHz, 20 ms one-way extra to reach cloud
    # ctrl = OJSTR(cloud_net_oneway_ms=20.0, cloud_cpu_capacity_ghz=500.0)

    # # Edges (GHz)
    # e0 = ctrl.add_edge(cpu_capacity_ghz=5.0)  # 5 GHz
    # e1 = ctrl.add_edge(cpu_capacity_ghz=3.0)  # 3 GHz

    # # Services (Gcycle per invocation, deadline in **ms**)
    # sA = ctrl.add_service(cycles_per_invocation_gcyc=1.0, deadline_ms=800.0)  # 1 Gcycle, 800 ms
    # sB = ctrl.add_service(cycles_per_invocation_gcyc=0.4, deadline_ms=500.0)  # 0.4 Gcycle, 500 ms

    # # Devices: uplink delays in **ms**
    # d0 = ctrl.add_device(service_id=sA, uplink_delay_to_edge_ms={e0: 20.0, e1: 50.0})
    # d1 = ctrl.add_device(service_id=sB, uplink_delay_to_edge_ms={e0: 40.0, e1: 30.0})

    # # --- Final allocation snapshot (persistent plan; devices reuse it over time) ---
    # plan = ctrl.compute_final_allocation()
    # # Print using your logger
    # from utils.logging import print

    # print(plan)
