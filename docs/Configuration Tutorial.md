## Tutorial: How to Write a Configuration File

This tutorial explains how to write a complete configuration file for C2TC simulations. A JSON scenario config file describes:

1. The physical environment
2. Applications
3. Processes
4. Topology and agents
5. Link types and network structure

Each part is parsed using the functions in `context.py`.

---

### 1. Environment

In your config file, define the environment and (optionally) the deployment:

```json
"environment": "environments/env1.json",
"deployment": "deployments/env1.json",
```

These are loaded by `load_environment()`. The deployment file contains a list of positions and transmission radii for each node.

---

### 2. Applications

Applications define processing requirements and are linked to benchmarks describing execution delays.

You can define applications in three ways:

* Embedded directly in the config file:

  ```json
  "applications": [
    {
      "name": "app0",
      "benchmark": { "distribution": { "type": "file", "data": "datasets/benchmark/app0.json" }, "cpu_ghz": 2.4 }
    }
  ]
  ```

* As array-style:

  ```json
  "applications_array": {
    "name": ["app0", "app1"],
    "benchmark": [
      { "distribution": { "type": "file", "data": "datasets/benchmark/app0.json" }, "cpu_ghz": 2.4 },
      { "distribution": { "type": "file", "data": "datasets/benchmark/app1.json" }, "cpu_ghz": 1.8 }
    ]
  }
  ```

  This is parsed into individual applications using indexed keys.

* From an external file:

  ```json
  "applications_import": "apps.json"
  ```

Each benchmark file typically contains a discrete probability distribution (PDF) of execution delays.

---

### 3. Processes

Processes define task instances that run in the environment. Each process references an application and can be defined:

* As an object list:

  ```json
  "processes": [ {...}, {...} ]
  ```

* As array-style (column-based representation):

  ```json
  "processes_array": {
    "name": ["P0", "P1"],
    "application": ["app0", "app1"],
    "mns": [1, 3],
    "max_delay_ms": [100, 200],
    "min_reliability": [0.99, 0.95],
    "aoi": [
      { "type": "circle", "data": [[50, 50], 20] },
      { "type": "line", "data": [[30, 30], [70, 70]] }
    ]
  }
  ```

  Each array is aligned by index to construct complete process objects.

* By importing from a file:

  ```json
  "processes_import": "procs.json"
  ```

The AoI (Area of Interest) field supports shapes like `circle` and `line`, interpreted by the `AoI` class in the framework.

---

### 4. Topology: Agents

Agents define the nodes involved in computation and routing.

```json
"agents": {
  "P0": { "type": "process" },
  "BR0": { "type": "host", "cpu_ghz": 2.0, "ram_gb": 8, "infinite_parallelism": false, "qtime_dist_function": "resourceallocation.tsch.sddu_qtime.qtime(g=4)" },
  "GW": { "type": "router" },
  "CN": { "type": "host", "cpu_ghz": 19.2, "ram_gb": "inf", "infinite_parallelism": true, "qtime_dist_function": "..." }
}
```

Agents may be:

* `process`: a software task
* `host`: computing node with specific CPU and RAM specs
* `router`: forwarding-only node

`qtime_dist_function` defines the queuing model for non-infinite hosts.

---

### 5. Link Classes

Link classes determine the communication delay between nodes:

```json
"link_classes": {
  "TSCH": { "type": "constant", "data": 2.7 },
  "BB": { "type": "file", "data": "datasets/ethernet/experiment.json" }
}
```

* `constant`: fixed delay in ms
* `file`: sampled delay distribution stored in a JSON dataset

---

### 6. Network Types

The `network` section links agents based on a specific topology. Supported structures:

* `direct_unidir` / `direct_bidir`: one-to-one link
* `star_bidir`: a central node with symmetric edges
* `mesh_bidir`: full interconnection between two sets
* `wireless_unidir`: probabilistic link based on a success probability matrix

Example of a wireless link:

```json
{
  "type": "wireless_unidir",
  "link_class": "TSCH",
  "data": {
    "set_a": ["P0", "P1"],
    "set_b": ["BR0", "BR1"],
    "prob_matrix": "resourceallocation.compute_brcoverage_matrix.compute_brcoverage_matrix(channel_model='mobile6tisch-3.json')"
  }
}
```

The `prob_matrix` is **not precomputed**: it's dynamically generated at runtime using a Python expression (interpreted via `dynamic_execute()`) which can load channel models and evaluate node visibility.

This allows flexibility in modeling radio propagation using preconfigured `.json` channel models stored in `datasets/`.

---

This file is parsed and validated by the `load_context()` pipeline, invoking all the necessary `load_*` functions to prepare the simulation graph and all agents involved.
