# Copyright (C) 2025  Marco Pettorali

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import networkx as nx

from networking.entities import AoI, Application, Host, Process
from networking.environment import Environment
from utils.colors import brightness, random_color
from utils.dynamic_execute import dynamic_execute


@dataclass
class Context:
    config_file_content: dict
    hosts: List[Host]
    environment: Environment
    applications: List[Application]
    processes: List[Process]
    topology_graph: nx.DiGraph
    json_filename: str
    links: dict

    def draw(self, ax, colors=None, offsets=None):
        for i, br in enumerate(self.environment.deployment):
            br.label = f"{i}"

        # Draw the environment
        self.environment.draw(ax)

        for i, (process_name, process) in enumerate(self.processes.items()):
            process.aoi.draw(
                ax,
                f"$A_{{{int(process_name.split('P')[1])}}}$",
                offset=((0, 0) if offsets is None else offsets.get(i, (0, 0))),
                color=(None if colors is None else colors[i]),
            )


def load_configfile(filename):
    """
    Loads a configuration file in JSON format.

    Args:
        filename (str): Path to the configuration file.

    Returns:
        dict: The parsed configuration file as a dictionary.
    """
    with open(filename) as f:
        config_file = json.load(f)
    return config_file


def load_environment(config):
    """
    Loads the environment from the configuration file.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        Environment: The loaded environment object.

    Raises:
        ValueError: If the 'environment' key is not found in the configuration.
    """
    if "environment" not in config:
        raise ValueError("'environment' key not found in the config file")

    env = Environment.load_env_from_file(config["environment"])

    if "deployment" in config:
        if isinstance(config["deployment"], str):
            env.load_deployment_from_file(config["deployment"])
        elif isinstance(config["deployment"], list):
            file, key = config["deployment"]
            env.load_deployment_from_file(file, key)

    return env


def load_applications(config, context):
    """
    Loads applications from the configuration file.

    Args:
        config (dict): The configuration dictionary.
        context (dict): The simulation context.

    Returns:
        list: A list of Application objects.
    """
    applications_data = []
    if "applications" in config:
        applications_data = config["applications"]
    elif "applications_array" in config:
        applications_data = [
            {key: values[i] for key, values in config["applications_array"].items()}
            for i in range(len(next(iter(config["applications_array"].values()))))
        ]
    elif "applications_import" in config:
        with open(Path(context["json_filename"]).parent / config["applications_import"], "r") as f:
            applications_data = json.load(f)

    ret = []
    for app in applications_data:
        app = Application(**app)
        ret.append(app)

    return ret


def load_processes(config, context):
    """
    Loads processes from the configuration file.

    Args:
        config (dict): The configuration dictionary.
        context (dict): The simulation context.

    Returns:
        list: A list of Process objects.

    Raises:
        ValueError: If a referenced application is not found in the context.
    """
    processes_data = []
    if "processes" in config:
        processes_data = config["processes"]
    elif "processes_array" in config:
        processes_data = [
            {key: values[i] for key, values in config["processes_array"].items()}
            for i in range(len(next(iter(config["processes_array"].values()))))
        ]
    elif "processes_import" in config:
        with open(Path(context["json_filename"]).parent / config["processes_import"], "r") as f:
            processes_data = json.load(f)

    ret = []
    for process in processes_data:
        if isinstance(process["application"], str):
            process["application"] = next(
                (app for app in context["applications"] if app.name == process["application"]),
                None,
            )
            if process["application"] is None:
                raise ValueError(f"Application {process['applications']} not found in the context")

        process = Process(**process)
        process.aoi.adapt_to_env(context["environment"])
        ret.append(process)

    return ret


def load_topology(config_file, context):
    """
    Loads the topology graph from the configuration file.

    Args:
        config_file (dict): The configuration dictionary.
        context (dict): The simulation context.

    Returns:
        nx.DiGraph: The topology graph.
    """
    topology = config_file["topology"]
    graph = nx.DiGraph()

    # Load nodes
    type_to_id, current_id, last_type = {}, -1, None
    for i, agent_name in enumerate(topology["agents"]):
        agent = topology["agents"][agent_name]
        agent_type = agent["type"]
        if agent_type != last_type:
            current_id += 1
            type_to_id[agent_type] = current_id
        graph.add_node(agent_name, agent=agent, subset=str(type_to_id[agent_type]), type=f"{agent_type}")
        last_type = agent_type

    # Load links
    for network_link in topology["network"]:
        weight = 1 if not "weight" in network_link else network_link["weight"]
        if network_link["type"].startswith("direct"):
            source, dest = network_link["data"]
            graph.add_edge(
                source,
                dest,
                link_class=network_link["link_class"],
                desc=topology["link_classes"][network_link["link_class"]],
                weight=weight,
            )
            if "bidir" in network_link["type"]:
                graph.add_edge(
                    dest,
                    source,
                    link_class=network_link["link_class"],
                    desc=topology["link_classes"][network_link["link_class"]],
                    weight=weight,
                )
        elif network_link["type"].startswith("star"):
            center, nodes = network_link["data"]["center"], network_link["data"]["set"]
            for node in nodes:
                if "inward" in network_link["type"] or "bidir" in network_link["type"]:
                    graph.add_edge(
                        node,
                        center,
                        link_class=network_link["link_class"],
                        desc=topology["link_classes"][network_link["link_class"]],
                        weight=weight,
                    )
                if "outward" in network_link["type"] or "bidir" in network_link["type"]:
                    graph.add_edge(
                        center,
                        node,
                        link_class=network_link["link_class"],
                        desc=topology["link_classes"][network_link["link_class"]],
                        weight=weight,
                    )
        elif network_link["type"].startswith("mesh"):
            set_a, set_b = network_link["data"]["set_a"], network_link["data"]["set_b"]
            for a in set_a:
                for b in set_b:
                    graph.add_edge(
                        a,
                        b,
                        link_class=network_link["link_class"],
                        desc=topology["link_classes"][network_link["link_class"]],
                        weight=weight,
                    )
                    if "bidir" in network_link["type"]:
                        graph.add_edge(
                            b,
                            a,
                            link_class=network_link["link_class"],
                            desc=topology["link_classes"][network_link["link_class"]],
                            weight=weight,
                        )
        elif network_link["type"].startswith("wireless"):
            set_a, set_b = network_link["data"]["set_a"], network_link["data"]["set_b"]
            prob_matrix = network_link["data"]["prob_matrix"]
            if isinstance(prob_matrix, str):
                prob_matrix = dynamic_execute(prob_matrix, context=context)

            for a in set_a:
                for b in set_b:
                    graph.add_edge(
                        a,
                        b,
                        link_class=network_link["link_class"],
                        desc=topology["link_classes"][network_link["link_class"]],
                        probability=prob_matrix[a][b],
                        weight=weight,
                    )
                    if "bidir" in network_link["type"]:
                        graph.add_edge(
                            b,
                            a,
                            link_class=network_link["link_class"],
                            desc=topology["link_classes"][network_link["link_class"]],
                            probability=prob_matrix[b][a],
                            weight=weight,
                        )
        else:
            raise ValueError(f"Unknown link type {network_link['type']}")

    return graph


def load_hosts(config_file, context):
    """
    Loads hosts from the configuration file.

    Args:
        config_file (dict): The configuration dictionary.
        context (dict): The simulation context.

    Returns:
        list: A list of Host objects.
    """
    hosts = [
        Host(h_name, h["cpu_ghz"], h["ram_gb"], h["infinite_parallelism"], h["qtime_dist_function"])
        for h_name, h in config_file["topology"]["agents"].items()
        if h["type"] == "host" and h["cpu_ghz"] > 0
    ]
    return hosts


def load_context(filename):
    """
    Loads the entire simulation context from a configuration file.

    Args:
        filename (str): Path to the configuration file.

    Returns:
        Context: The loaded simulation context.
    """
    context = {}

    config_file = load_configfile(filename)
    context["config_file_content"] = config_file
    context["json_filename"] = filename

    env = load_environment(config_file)
    context["environment"] = env

    applications = load_applications(config_file, context)
    context["applications"] = applications

    processes = load_processes(config_file, context)
    context["processes"] = {}
    for process in processes:
        context["processes"][process.name] = process

    topology_graph = load_topology(config_file, context)
    context["topology_graph"] = topology_graph

    hosts = load_hosts(config_file, context)
    context["hosts"] = {}
    for h in hosts:
        context["hosts"][h.label] = h

    context["links"] = {}

    context = Context(**context)
    return context
