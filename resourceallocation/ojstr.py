"""
OJSTR (simplified, deadline-driven) — Assumptions & Adaptation Notes
====================================================================

This implementation adapts "Online Joint Service placement, Task scheduling,
and Resource allocation" (OJSTR) to a minimal, cost-free / energy-free setting
focused on meeting per-task deadlines and maximizing the number of tasks served.

A) OJSTR Paper Assumptions (conceptual model)
---------------------------------------------
- Compute time on device/edge is modeled via CPU capacity:
  time = cycles / cpu_rate.
- Uplink transmission is modeled through radio rates (and power), so upload
  time depends on task size and channel conditions.
- Optimization includes cost components (e.g., device energy, cloud tenancy).
- Cloud is not assumed “free by default” in the cost model (tenancy may apply).
- Service placement may be reconfigured over time (subject to constraints).

B) Our Adaptation Assumptions (differences vs paper)
----------------------------------------------------
- No monetary costs and no energy models: entirely removed.
- Cloud compute is assumed abundant; its compute time is neglected. An optional
  constant network delay `cloud_rtt_s` can be used when offloading to cloud.
TODO - Device/edge compute time IS modeled: time = cycles / cpu_rate (kept as in OJSTR).
- Uplink is modeled as a **fixed delay** provided per (device, edge) pair:
  `uplink_delay_to_edge_s[edge_id]` in seconds. This same best-edge uplink is
  used as access path to the cloud (plus `cloud_rtt_s` if set).
TODO - Task size is accepted by the API for compatibility, but **ignored** for uplink;
  upload time does **not** scale with bits because you directly provide the delay.
- **Placement persistence (add-only)**: once a service is placed on an edge,
  it CANNOT be deallocated nor migrated to another host. Placement is monotonic
  (we only add new services if there is remaining storage capacity). This is a
  stricter policy than typical OJSTR reconfiguration and is enforced here.



D. Placement (per-edge 0-1 Knapsack)
------------------------------------
- Value(service, edge) = estimated number of currently queued tasks (across
  devices) of that service that COULD meet the deadline at that edge, given
  the provided uplink delays and the edge CPU speed.
- Weight = service image size (MB). Capacity = edge storage capacity (MB).
- We pick the set of services maximizing that estimated feasible demand.

E. Scheduling (Greedy, Min Completion Time)
-------------------------------------------
- For each device with a head-of-line (HoL) task, enumerate feasible options
  {local, any edge with the service placed, cloud}. Each option has a total
  completion time:
    - local: `cycles/device_cpu`
    - edge:  `uplink_delay + cycles/edge_cpu`
    - cloud: `best_uplink_delay + cloud_rtt_s`
- Pick the option with the MINIMUM completion time while respecting CPU
  budgets for the current slot. If none is feasible, the task is deferred.

F. Dynamics / Online Operations
-------------------------------
- `add_service(...)`: dynamically add a new service/application.
- `add_device(...)`:  dynamically add a new device, providing its
  `uplink_delay_to_edge_s` mapping.
- `add_task(device, service, size_bits)`: dynamically enqueue a request
  (size is currently ignored — delay is directly provided per device-edge).

G. Optional Network Constant
----------------------------
- `cloud_rtt_s` (default 0.0): extra network delay when sending to the cloud
  (e.g., routing/backhaul/round-trip). Set > 0 if you want to penalize cloud.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math
from collections import deque


# ===========================
# Data models (no costs, no energy)
# ===========================

@dataclass
class Service:
    """
    Service (i.e., 'application') type.

    Attributes
    ----------
    id : int
        Unique service id.
    name : str
        Human-readable name.
    store_size_mb : int
        Storage footprint if placed on an edge node (knapsack weight).
    cycles_per_task : float
        CPU cycles required to execute one task of this service.
    deadline_s : float
        Per-task deadline (seconds). Scheduling must meet it per task.
    """
    id: int
    name: str
    store_size_mb: int
    cycles_per_task: float
    deadline_s: float


@dataclass
class Device:
    """
    End device producing tasks.

    Attributes
    ----------
    id : int
        Unique device id.
    cpu_cycles_per_s : float
        Local CPU capacity (cycles/second).
    uplink_delay_to_edge_s : Dict[int, float]
        Map edge_id -> *fixed uplink delay in seconds* to reach that edge.
        (Provided by you; replaces any rate-based calculation.)
    queue : deque
        FIFO of pending tasks (service_id, size_bits). size_bits accepted
        for compatibility, but currently unused in time computations.
    """
    id: int
    cpu_cycles_per_s: float
    uplink_delay_to_edge_s: Dict[int, float] = field(default_factory=dict)
    queue: deque = field(default_factory=deque)


@dataclass
class EdgeNode:
    """
    Edge node (BS/server).

    Attributes
    ----------
    id : int
        Unique edge id.
    cpu_cycles_per_s : float
        Edge CPU capacity (cycles/second).
    storage_capacity_mb : int
        Storage capacity for service images (knapsack capacity).
    placed_services : set
        Set of service ids currently placed (available this slot).
    """
    id: int
    cpu_cycles_per_s: float
    storage_capacity_mb: int
    placed_services: set = field(default_factory=set)


@dataclass
class Cloud:
    """
    Cloud node (compute assumed abundant; only uplink delay + cloud_rtt_s matter).
    """
    cpu_cycles_per_s: float = 1e15  # effectively 'infinite' here


@dataclass
class OJSTRParams:
    """
    Controller-level parameters (no costs, no energy).

    Attributes
    ----------
    slot_duration_s : float
        Slot duration in seconds (scheduling/allocation window).
    cloud_rtt_s : float
        Extra network delay term when offloading to cloud (default 0.0).
    """
    slot_duration_s: float = 1.0
    cloud_rtt_s: float = 0.0


# ===========================
# Core controller (no costs/energy)
# ===========================

class OJSTRController:
    """
    OJSTR-like online controller (simplified, cost/energy removed; delays provided directly).

    Per slot:
      1) Service placement (per-edge knapsack) to maximize the number of
         potentially-feasible tasks that could meet deadlines at that edge.
      2) Task scheduling + resource allocation picking the option with
         the smallest completion time (local/edge/cloud) while respecting CPU budgets.

    Dynamics supported:
      - add_service, add_device, add_task (online arrivals).
    """

    def __init__(self, params: OJSTRParams):
        self.params = params
        self.services: Dict[int, Service] = {}
        self.devices: Dict[int, Device] = {}
        self.edges: Dict[int, EdgeNode] = {}
        self.cloud = Cloud()

        # Book-keeping
        self._next_service_id = 0
        self._next_device_id = 0
        self._next_edge_id = 0

        # Last-slot report
        self.last_slot_log: Dict[str, object] = {}

    # ------------- Dynamic registry -------------

    def add_edge(self, cpu_cycles_per_s: float, storage_capacity_mb: int) -> int:
        eid = self._next_edge_id
        self._next_edge_id += 1
        self.edges[eid] = EdgeNode(id=eid, cpu_cycles_per_s=cpu_cycles_per_s, storage_capacity_mb=storage_capacity_mb)
        return eid

    def add_service(self, name: str, store_size_mb: int,
                    cycles_per_task: float, deadline_s: float) -> int:
        """
        Dynamically add a new Service (a.k.a. 'application').
        """
        sid = self._next_service_id
        self._next_service_id += 1
        self.services[sid] = Service(
            id=sid,
            name=name,
            store_size_mb=store_size_mb,
            cycles_per_task=cycles_per_task,
            deadline_s=deadline_s
        )
        return sid

    def add_device(self, cpu_cycles_per_s: float,
                   uplink_delay_to_edge_s: Dict[int, float]) -> int:
        """
        Dynamically add a new device with its *delays* toward edges (seconds).
        Example: {edge_id_0: 0.015, edge_id_1: 0.030}
        """
        did = self._next_device_id
        self._next_device_id += 1
        self.devices[did] = Device(
            id=did,
            cpu_cycles_per_s=cpu_cycles_per_s,
            uplink_delay_to_edge_s=dict(uplink_delay_to_edge_s),
        )
        return did

    # ----------------- Native online operation -----------------

    def add_task(self, device_id: int, service_id: int, size_bits: int) -> None:
        """
        Add a new task request dynamically (device -> service).
        NOTE: size_bits is currently ignored, since uplink delay is provided directly.
        """
        self.devices[device_id].queue.append((service_id, size_bits))

    # ===========================
    # Slot routine
    # ===========================

    def step(self) -> Dict:
        """
        One time-slot decision:
          (1) Service placement (per edge), knapsack-like by "feasible demand"
          (2) Task scheduling + resource allocation (device/self, edge, or cloud)
        Returns a per-slot report.
        """
        # 1) Placement
        self._service_placement_knapsack_all_edges()

        # 2) Scheduling + resource allocation (deadline-feasible, min completion time)
        result_sched = self._schedule_and_allocate()

        # Prepare report
        report = {
            "served_tasks": result_sched["served_tasks"],
            "dropped_or_deferred": result_sched["dropped_or_deferred"],
            "edge_cpu_used": result_sched["edge_cpu_used"],
            "device_cpu_used": result_sched["device_cpu_used"],
        }
        self.last_slot_log = report
        return report

    # ===========================
    # (1) Service placement (per-edge knapsack)
    # ===========================

    def _service_placement_knapsack_all_edges(self) -> None:
        """
        For each edge, add (if space permits) new services that maximize the
        estimated number of currently queued tasks that could meet deadlines at that edge.
        NOTE: Placement is PERSISTENT (add-only): already placed services remain placed
        and are never removed or migrated.
        """
        for e in self.edges.values():
            # Compute remaining storage capacity (MB) considering already-placed services
            used_mb = 0
            for sid in e.placed_services:
                used_mb += self.services[sid].store_size_mb
            remaining_mb = max(e.storage_capacity_mb - used_mb, 0)

            # Build candidate items ONLY for services NOT yet placed on this edge
            items: List[Tuple[int, int, float]] = []  # (service_id, weight_MB, value=feasible_count)
            if remaining_mb > 0:
                for sid, s in self.services.items():
                    if sid in e.placed_services:
                        continue  # already placed -> persistent, skip as candidate

                    feasible_count = 0
                    # Count how many queued tasks (across devices) of this service
                    # could meet deadline if sent to this edge
                    for d in self.devices.values():
                        delay_to_e = d.uplink_delay_to_edge_s.get(e.id, math.inf)
                        if delay_to_e == math.inf or not d.queue:
                            continue
                        for (req_sid, _size_bits) in d.queue:
                            if req_sid != sid:
                                continue
                            edge_exec_time = s.cycles_per_task / e.cpu_cycles_per_s
                            total_time = delay_to_e + edge_exec_time
                            if total_time <= s.deadline_s:
                                feasible_count += 1

                    if feasible_count > 0:
                        items.append((sid, s.store_size_mb, float(feasible_count)))

            # Solve 0-1 knapsack by DP on the REMAINING capacity
            chosen = self._knapsack_01_dp(items, remaining_mb)

            # Persistent placement: ADD (do not replace) chosen services
            for sid in chosen:
                e.placed_services.add(sid)


    @staticmethod
    def _knapsack_01_dp(items: List[Tuple[int, int, float]], capacity: int) -> List[int]:
        """
        Simple 0-1 knapsack: items = (id, weight, value_score), capacity in MB.
        Returns the set of chosen item ids maximizing total score.
        """
        n = len(items)
        if n == 0 or capacity <= 0:
            return []

        dp = [0.0] * (capacity + 1)
        keep = [[False] * (capacity + 1) for _ in range(n)]

        for i, (_id, w, v) in enumerate(items):
            if w > capacity:
                continue
            for c in range(capacity, w - 1, -1):
                if dp[c - w] + v > dp[c]:
                    dp[c] = dp[c - w] + v
                    keep[i][c] = True

        chosen: List[int] = []
        c = capacity
        for i in range(n - 1, -1, -1):
            if keep[i][c]:
                chosen.append(items[i][0])
                c -= items[i][1]
        return chosen

    # ===========================
    # (2) Scheduling + resource allocation
    # ===========================

    def _schedule_and_allocate(self) -> Dict:
        """
        Greedy matching-style routine:
        - iterate devices that have a head-of-line (HoL) task;
        - compute feasible options: local, any edge with service placed, cloud;
        - pick the option with MINIMUM completion time (total_time),
          while respecting per-slot CPU budgets at device and edges.

        Returns per-slot accounting.
        """
        T = self.params.slot_duration_s

        # Per-node CPU budgets in cycles (for this slot)
        device_budget = {d.id: d.cpu_cycles_per_s * T for d in self.devices.values()}
        edge_budget = {e.id: e.cpu_cycles_per_s * T for e in self.edges.values()}

        served_tasks = []
        dropped_or_deferred = []

        # Build candidates: devices with a HoL task
        candidates: List[Tuple[int, int, int, float]] = []
        for d in self.devices.values():
            if len(d.queue) == 0:
                continue
            sid, size_bits = d.queue[0]
            s = self.services[sid]
            candidates.append((d.id, sid, size_bits, s.deadline_s))

        # Helper: best feasible completion time for sorting
        def best_feasible_total_time(dev_id: int, sid: int, size_bits: int) -> float:
            options = self._enumerate_options(dev_id, sid,
                                              device_budget, edge_budget)
            if not options:
                return math.inf
            return min(opt["total_time"] for opt in options)

        # Sort by smallest best feasible completion time (if none, goes to the end)
        candidates.sort(key=lambda t: best_feasible_total_time(t[0], t[1], t[2]))

        # Serve in order
        for dev_id, sid, size_bits, _dl in candidates:
            options = self._enumerate_options(dev_id, sid,
                                              device_budget, edge_budget)
            if not options:
                dropped_or_deferred.append((dev_id, sid))
                continue

            # pick the option with minimum completion time
            choice = min(options, key=lambda o: o["total_time"])

            # consume resources and pop HoL
            s = self.services[sid]
            if choice["where"] == "local":
                device_budget[dev_id] -= s.cycles_per_task
            elif choice["where"] == "edge":
                edge_budget[choice["edge_id"]] -= s.cycles_per_task
            else:  # "cloud"
                pass  # no CPU budget tracked for cloud

            self.devices[dev_id].queue.popleft()
            served_tasks.append((dev_id, sid, choice["where"], choice.get("edge_id")))

        return {
            "served_tasks": served_tasks,
            "dropped_or_deferred": dropped_or_deferred,
            "edge_cpu_used": {e.id: e.cpu_cycles_per_s * T - edge_budget[e.id] for e in self.edges.values()},
            "device_cpu_used": {d.id: d.cpu_cycles_per_s * T - device_budget[d.id] for d in self.devices.values()},
        }

    def _enumerate_options(self, dev_id: int, service_id: int,
                           device_budget: Dict[int, float],
                           edge_budget: Dict[int, float]) -> List[Dict]:
        """
        Return all feasible options with completion time for (dev, service, task).
        Feasibility is checked against per-task deadline and remaining CPU budgets.
        No costs, no energy; we choose the option with MINIMUM total_time.
        """
        d = self.devices[dev_id]
        s = self.services[service_id]
        cloud_rtt = self.params.cloud_rtt_s

        options: List[Dict] = []

        # --- Local execution ---
        local_exec_time = s.cycles_per_task / d.cpu_cycles_per_s
        if local_exec_time <= s.deadline_s and device_budget[dev_id] >= s.cycles_per_task:
            options.append({"where": "local", "total_time": local_exec_time})

        # --- Edge execution (any edge with the service placed) ---
        for e in self.edges.values():
            if service_id not in e.placed_services:
                continue
            delay_to_e = d.uplink_delay_to_edge_s.get(e.id, math.inf)
            if delay_to_e == math.inf:
                continue

            edge_exec_time = s.cycles_per_task / e.cpu_cycles_per_s
            total_time = delay_to_e + edge_exec_time
            if total_time <= s.deadline_s and edge_budget[e.id] >= s.cycles_per_task:
                options.append({"where": "edge", "edge_id": e.id, "total_time": total_time})

        # --- Cloud execution ---
        # Use best available (device->edge) uplink delay as access path to the cloud.
        best_uplink = min(d.uplink_delay_to_edge_s.values(), default=math.inf)
        cloud_time = best_uplink + cloud_rtt
        if cloud_time <= s.deadline_s:
            options.append({"where": "cloud", "total_time": cloud_time})

        return options


# ===========================
# Minimal usage example (delays-based)
# ===========================
if __name__ == "__main__":
    params = OJSTRParams(slot_duration_s=1.0, cloud_rtt_s=0.02)  # e.g., 20 ms extra when using cloud
    ctrl = OJSTRController(params)

    # Edges
    e0 = ctrl.add_edge(cpu_cycles_per_s=5e9, storage_capacity_mb=200)
    e1 = ctrl.add_edge(cpu_cycles_per_s=3e9, storage_capacity_mb=150)

    # Services (no costs, no energy)
    sA = ctrl.add_service(name="VideoAnalytics", store_size_mb=60,
                          cycles_per_task=1.0e9, deadline_s=0.8)
    sB = ctrl.add_service(name="AnomalyDetect", store_size_mb=40,
                          cycles_per_task=4.0e8, deadline_s=0.5)

    # Devices (provide *delays* in seconds to each edge)
    d0 = ctrl.add_device(cpu_cycles_per_s=2e9,
                         uplink_delay_to_edge_s={e0: 0.02, e1: 0.05})
    d1 = ctrl.add_device(cpu_cycles_per_s=1e9,
                         uplink_delay_to_edge_s={e0: 0.04, e1: 0.03})

    # Dynamic arrivals (size_bits ignored here)
    ctrl.add_task(d0, sA, size_bits=8_000_000)
    ctrl.add_task(d0, sB, size_bits=4_000_000)
    ctrl.add_task(d1, sA, size_bits=8_000_000)

    # Run a few slots
    for t in range(3):
        rep = ctrl.step()
        print(f"[slot {t}] report:", rep)
