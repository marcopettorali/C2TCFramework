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

import tabulate
from rich.text import Text

from networking.entities import Application, Host, Process, ProcessSplit
from utils.colors import colorize_text, heatmap
from utils.printing import print


class AllocationMap:
    def __init__(self):
        self._allocated_process_splits = {}
        self._hosts = {}

        self._allocation_map = {}
        self._host_available_utilization = {}
        self._available_ram_gb = {}

        self._split_labels_map = {}

    def set_hosts(self, hosts):
        host: Host
        for host_label in hosts:
            host: Host = hosts[host_label]
            self._hosts[host.label] = host
            self._allocation_map[host.label] = []
            self._host_available_utilization[host.label] = 1
            self._available_ram_gb[host.label] = host.ram_gb

    def get_allocated_process_splits(self):
        return self._allocated_process_splits

    def _allocate_debug(self, process_split: ProcessSplit, host_label, cpu_share, ram_occupied):
        assert isinstance(process_split, ProcessSplit), f"process_split must be of type ProcessSplit, got {type(process_split)}"
        self._allocated_process_splits[process_split.split_name] = process_split
        self._allocation_map[host_label].append((process_split.split_name, cpu_share))
        self._host_available_utilization[host_label] -= cpu_share
        self._available_ram_gb[host_label] -= ram_occupied

    def get_process_split_names_on_host_for_process(self, process_name, host_label):
        return [p for p, _ in self._allocation_map[host_label] if self._allocated_process_splits[p].parent_process_name == process_name]

    def allocate(self, process_split: ProcessSplit, host_label, cpu_share, merge_splits_if_already_allocated=False):
        """
        Allocate a process split to a host with a given CPU share.
        Returns True if allocate() caused a new process to be allocated, False otherwise.
        Note that if the process is merged instead, the function will return False.
        """
        assert host_label in self._allocation_map, f"Host {host_label} not found in the allocation map"

        host: Host = self._hosts[host_label]

        # check if the host has other splits of the same process
        if merge_splits_if_already_allocated:
            print(f"Checking if the process {process_split.parent_process_name} is already allocated on {host_label}")
            # if there are other splits of the same process, merge them
            # by not adding the new split to the allocation map, but by increasing the CPU share of the existing split
            # Note: the RAM does not vary
            splits_on_host = self.get_process_split_names_on_host_for_process(process_split.parent_process_name, host_label)

            # assume only one split of the same process is allocated on the host
            assert (
                len(splits_on_host) <= 1
            ), f"Multiple splits of the same process {process_split.parent_process_name} are allocated on the host {host_label} and not currently supported"

            # merge the splits by increasing the CPU share
            if len(splits_on_host) == 1:
                print(f"Process {process_split.parent_process_name} is already allocated on {host_label}")

                p_part_name = splits_on_host[0]
                self._allocated_process_splits[p_part_name].mns += process_split.mns
                self.update_cpu_share(p_part_name, delta_cpu_share=cpu_share)
                if not host.infinite_parallelism:
                    if self._host_available_utilization[host_label] < cpu_share:

                        if abs(self._host_available_utilization[host_label] - cpu_share) < 1e-6:
                            print(
                                f"Warning: requested CPU share {cpu_share} is higher than available CPU {self._host_available_utilization[host_label]} but the difference is negligible",
                                style="warning",
                            )
                        else:
                            raise Exception(
                                f"Host {host_label} does not have enough CPU to allocate {cpu_share}, available CPU = {self._host_available_utilization[host_label]}"
                            )

                    self._host_available_utilization[host_label] -= cpu_share
                    print(f"Available CPU on {host_label} after merging: {self._host_available_utilization[host_label]}")
                print(self.to_rich_text())
                return False

        # check if the host has enough CPU to allocate the process
        if not host.infinite_parallelism and cpu_share > self._host_available_utilization[host_label]:
            raise Exception(
                f"Host {host_label} does not have enough CPU to allocate {cpu_share}, available CPU = {self._host_available_utilization[host_label]}"
            )

        # check if the host has enough RAM to allocate the process
        process_app: Application = process_split.application
        if process_app.ram_occupancy_gb > self._available_ram_gb[host_label]:
            raise Exception(
                f"Host {host_label} does not have enough RAM to allocate {process_split.name} (required RAM = {process_app.ram_occupancy_gb} GB, available RAM = {self._available_ram_gb[host_label]} GB)"
            )

        # allocate the process
        self._allocation_map[host_label].append((process_split.split_name, cpu_share))

        # add the process to the allocated processes
        self._allocated_process_splits[process_split.split_name] = process_split

        # update the available utilization (if the host has not infinite parallelism)
        if not host.infinite_parallelism:
            self._host_available_utilization[host_label] -= cpu_share

        # update the available RAM
        self._available_ram_gb[host_label] -= process_app.ram_occupancy_gb

        print(
            f"Allocation map after allocating {process_split.split_name} on {host_label} with CPU share {cpu_share}",
            style="debug",
        )
        print(self.to_rich_text())

        return True

    def get_allocated_processes_names_on_host(self, host_label):
        return self._allocation_map[host_label]

    def get_host_label_for_process(self, process_name):
        for host_label, processes in self._allocation_map.items():
            for p, _ in processes:
                if p == process_name:
                    return host_label
        return None

    def get_host_and_cpushare_for_process(self, process_name):
        for host_label, processes in self._allocation_map.items():
            for p, cpushare in processes:
                if p == process_name:
                    return host_label, cpushare
        return None, None

    def update_cpu_share(self, process_name, new_cpu_share=None, delta_cpu_share=None):
        """
        Update the CPU share of a process.
        If new_cpu_share is provided, the CPU share is set to this value.
        If delta_cpu_share is provided, the CPU share is increased by this value.
        """
        host_label, cpu_share = self.get_host_and_cpushare_for_process(process_name)
        if host_label is None:
            raise Exception(f"Process {process_name} not found in the allocation map")

        if new_cpu_share is not None:
            # update the CPU share
            self._allocation_map[host_label] = [(p, c) if p != process_name else (p, new_cpu_share) for p, c in self._allocation_map[host_label]]
            return

        if delta_cpu_share is not None:
            # update the CPU share
            self._allocation_map[host_label] = [(p, c) if p != process_name else (p, c + delta_cpu_share) for p, c in self._allocation_map[host_label]]
            return

        raise Exception("Either new_cpu_share or delta_cpu_share must be provided")

    def get_available_utilization(self, host_label):
        return (
            0
            if self._host_available_utilization[host_label] < 1e-3
            else 1 if self._hosts[host_label].infinite_parallelism else self._host_available_utilization[host_label]
        )

    def get_available_ram_gb(self, host_label):
        return self._available_ram_gb[host_label]

    def add_process_split_labels_map(self, process_name, split_name):
        # check if the process has already been added to the split labels map
        if process_name not in self._split_labels_map:
            self._split_labels_map[process_name] = []

        # add the split label to the process
        self._split_labels_map[process_name].append(split_name)

    def get_splits_name_for_process(self, process_name):
        """
        Get the split labels for the given process. If the process has no splits, return None.
        """
        if process_name not in self._split_labels_map:
            return None
        return self._split_labels_map[process_name]

    def to_rich_text(self):

        # get all free CPU and RAM values
        free_cpus = [
            (
                self._host_available_utilization[host_label] * self._hosts[host_label].cpu_ghz
                if host_label != "CN"
                else 1.1 * max([self._host_available_utilization[y] * self._hosts[y].cpu_ghz for y in self._allocation_map if y != "CN"])
            )
            for host_label in self._allocation_map
        ]
        free_cpus_colors = heatmap(free_cpus, ["red", "yellow", "green"])

        free_ram = [
            (self._available_ram_gb[host_label] if host_label != "CN" else 1.1 * max([self._available_ram_gb[y] for y in self._allocation_map if y != "CN"]))
            for host_label in self._allocation_map
        ]

        free_rams_colors = heatmap(free_ram, ["red", "yellow", "green"])

        cpu_labels = [
            f"{self._host_available_utilization[host_label] * self._hosts[host_label].cpu_ghz :.2f} / {self._hosts[host_label].cpu_ghz :.2f} GHz"
            for i, host_label in enumerate(self._allocation_map)
        ]
        ram_labels = [
            f"{self._available_ram_gb[host_label] :.2f} / {self._hosts[host_label].ram_gb :.2f} GB" for i, host_label in enumerate(self._allocation_map)
        ]

        data = []
        for i, (host_label, processes) in enumerate(self._allocation_map.items()):
            line = [f"{host_label}{'*' if self._hosts[host_label].infinite_parallelism else ''}"]

            proc_info = "\n".join(
                f"({self._allocated_process_splits[process_name].split_name}: MNs={self._allocated_process_splits[process_name].mns}, CPU share={cpu_share})"
                for process_name, cpu_share in processes
            )
            line.append(proc_info)

            # append available CPU and RAM
            line.append(colorize_text(cpu_labels[i], free_cpus_colors[i]))
            line.append(colorize_text(ram_labels[i], free_rams_colors[i]))

            data.append(line)

        ret = tabulate.tabulate(
            data,
            headers=["Host", "Process", "Free / Total CPU", "Free / Total RAM"],
            tablefmt="rounded_grid",
            colalign=["right", "left", "right", "right"],
            disable_numparse=True,
        )

        return Text.from_ansi(ret)
