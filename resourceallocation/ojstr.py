from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
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
    uplink_rate_to_edge_mbps : Dict[int, float]
        Map edge_id -> achievable uplink rate (Mbit/s).
    queue : deque
        FIFO of pending tasks (service_id, size_bits).
    """
    id: int
    cpu_cycles_per_s: float
    uplink_rate_to_edge_mbps: Dict[int, float] = field(default_factory=dict)
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
    Cloud node (compute assumed abundant; only uplink time matters here).
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
    """
    slot_duration_s: float = 1.0


# ===========================
# Core controller (no costs/energy)
# ===========================

class OJSTRController:
    """
    OJSTR-like online controller (simplified, cost/energy removed).

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
                   uplink_rate_to_edge_mbps: Dict[int, float]) -> int:
        """
        Dynamically add a new device with its radio rates toward edges.
        """
        did = self._next_device_id
        self._next_device_id += 1
        self.devices[did] = Device(
            id=did,
            cpu_cycles_per_s=cpu_cycles_per_s,
            uplink_rate_to_edge_mbps=dict(uplink_rate_to_edge_mbps),
        )
        return did

    # ----------------- Native online operation -----------------

    def add_task(self, device_id: int, service_id: int, size_bits: int) -> None:
        """
        Add a new task request dynamically (device -> service).
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
        For each edge, decide which services to keep/place this slot.

        Value(service, edge) = estimated number of queued tasks that
        could meet their deadline at this edge given:
            - device->edge uplink rate
            - edge CPU speed and per-task cycles
            - task sizes currently in queues

        We then pick the subset of services (by storage capacity) maximizing
        the total estimated feasible demand at that edge.
        """
        for e in self.edges.values():
            # Build candidate items (service -> (weight, value_score))
            items: List[Tuple[int, int, float]] = []  # (service_id, weight_MB, value=feasible_count)
            for sid, s in self.services.items():
                feasible_count = 0
                # Count how many queued tasks (across devices) of this service
                # could meet deadline if sent to this edge
                for d in self.devices.values():
                    rate_mbps = d.uplink_rate_to_edge_mbps.get(e.id, 0.0)
                    if rate_mbps <= 0:
                        continue
                    if not d.queue:
                        continue
                    # Consider all queued tasks (coarse estimate)
                    for (req_sid, size_bits) in d.queue:
                        if req_sid != sid:
                            continue
                        uplink_time = (size_bits / 1e6) / rate_mbps  # seconds
                        edge_exec_time = s.cycles_per_task / e.cpu_cycles_per_s
                        total_time = uplink_time + edge_exec_time
                        if total_time <= s.deadline_s:
                            feasible_count += 1

                if feasible_count > 0 or sid in e.placed_services:  # allow inertia
                    items.append((sid, s.store_size_mb, float(feasible_count)))

            # Solve 0-1 knapsack by DP (capacity = e.storage_capacity_mb)
            chosen = self._knapsack_01_dp(items, e.storage_capacity_mb)

            # Update placement set (chosen is the list of service ids)
            e.placed_services = set(chosen)

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
        # cloud assumed abundant; only uplink time matters here

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
            options = self._enumerate_options(dev_id, sid, size_bits,
                                              device_budget, edge_budget)
            if not options:
                return math.inf
            return min(opt["total_time"] for opt in options)

        # Sort by smallest best feasible completion time (if none, goes to the end)
        candidates.sort(key=lambda t: best_feasible_total_time(t[0], t[1], t[2]))

        # Serve in order
        for dev_id, sid, size_bits, _dl in candidates:
            options = self._enumerate_options(dev_id, sid, size_bits,
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

    def _enumerate_options(self, dev_id: int, service_id: int, size_bits: int,
                           device_budget: Dict[int, float],
                           edge_budget: Dict[int, float]) -> List[Dict]:
        """
        Return all feasible options with completion time for (dev, service, task).
        Feasibility is checked against deadline and remaining budgets.
        No costs, no energy; we choose the option with MINIMUM total_time.
        """
        d = self.devices[dev_id]
        s = self.services[service_id]

        options: List[Dict] = []

        # --- Local execution ---
        local_exec_time = s.cycles_per_task / d.cpu_cycles_per_s
        if local_exec_time <= s.deadline_s and device_budget[dev_id] >= s.cycles_per_task:
            options.append({"where": "local", "total_time": local_exec_time})

        # --- Edge execution (any edge with the service placed + radio link) ---
        for e in self.edges.values():
            if service_id not in e.placed_services:
                continue
            rate_mbps = d.uplink_rate_to_edge_mbps.get(e.id, 0.0)
            if rate_mbps <= 0:
                continue

            uplink_time = (size_bits / 1e6) / rate_mbps  # seconds
            edge_exec_time = s.cycles_per_task / e.cpu_cycles_per_s
            total_time = uplink_time + edge_exec_time
            if total_time <= s.deadline_s and edge_budget[e.id] >= s.cycles_per_task:
                options.append({"where": "edge", "edge_id": e.id, "total_time": total_time})

        # --- Cloud execution ---
        # Assume cloud compute time negligible; completion time ~ uplink via best available radio.
        best_uplink = math.inf
        for e in self.edges.values():
            rate_mbps = d.uplink_rate_to_edge_mbps.get(e.id, 0.0)
            if rate_mbps > 0:
                uplink_time = (size_bits / 1e6) / rate_mbps
                best_uplink = min(best_uplink, uplink_time)
        if best_uplink < math.inf and best_uplink <= s.deadline_s:
            options.append({"where": "cloud", "total_time": best_uplink})

        return options


# ===========================
# Minimal usage example
# ===========================
if __name__ == "__main__":
    params = OJSTRParams(slot_duration_s=1.0)
    ctrl = OJSTRController(params)

    # Edges
    e0 = ctrl.add_edge(cpu_cycles_per_s=5e9, storage_capacity_mb=200)
    e1 = ctrl.add_edge(cpu_cycles_per_s=3e9, storage_capacity_mb=150)

    # Services (no costs, no energy)
    sA = ctrl.add_service(name="VideoAnalytics", store_size_mb=60,
                          cycles_per_task=1.0e9, deadline_s=0.8)
    sB = ctrl.add_service(name="AnomalyDetect", store_size_mb=40,
                          cycles_per_task=4.0e8, deadline_s=0.5)

    # Devices (radio links only)
    d0 = ctrl.add_device(cpu_cycles_per_s=2e9,
                         uplink_rate_to_edge_mbps={e0: 50.0, e1: 10.0})
    d1 = ctrl.add_device(cpu_cycles_per_s=1e9,
                         uplink_rate_to_edge_mbps={e0: 20.0, e1: 30.0})

    # Dynamic arrivals
    ctrl.add_task(d0, sA, size_bits=8_000_000)  # 1 MB
    ctrl.add_task(d0, sB, size_bits=4_000_000)
    ctrl.add_task(d1, sA, size_bits=8_000_000)

    # Run a few slots
    for t in range(3):
        rep = ctrl.step()
        print(f"[slot {t}] report:", rep)
