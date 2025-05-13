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

from dataclasses import dataclass

import shapely as sh

from utils.colors import brightness, random_color
from utils.distribution import DistributionDescriptor
from utils.geom import compute_coverage_area


@dataclass
class Benchmark:
    cpu_ghz: float
    distribution: DistributionDescriptor

    def __post_init__(self):
        if isinstance(self.distribution, DistributionDescriptor):
            return

        self.distribution = DistributionDescriptor(**self.distribution)


@dataclass
class Application:
    name: str
    ram_occupancy_gb: float
    benchmark: Benchmark

    def __post_init__(self):
        if isinstance(self.benchmark, Benchmark):
            return

        self.benchmark = Benchmark(**self.benchmark)


@dataclass
class AoI:
    type: str
    data: list

    polygon: sh.geometry.Polygon = None

    def __post_init__(self):
        if self.type == "polygon":
            self.polygon = sh.Polygon(self.data)
        elif self.type == "line":
            p1 = sh.Point(self.data[0])
            p2 = sh.Point(self.data[1])
            self.polygon = sh.LineString([p1, p2])
            # buffer line
            self.polygon = self.polygon.buffer(1)
        elif self.type == "point":
            self.polygon = sh.Point(self.data[0]).buffer(10)
        elif self.type == "circle":
            self.center = sh.Point(self.data[0])
            self.radius = self.data[1]
            self.polygon = self.center.buffer(self.radius)
        else:
            raise ValueError(f"Unknown aoi type: {self.type}")

        if self.polygon is None:
            raise ValueError(f"Unknown aoi type: {self.type}")

    def adapt_to_env(self, env):
        if self.type == "circle":
            self.polygon = compute_coverage_area(self.center, self.radius, env.obstacles, env.get_boundary_obstacle())

    def draw(self, ax, label, offset=None, color=None):
        text_point = list(self.polygon.centroid.coords[0])
        text_point[0] += offset[0] if offset else 0
        text_point[1] += offset[1] if offset else 0

        if not color:
            color = random_color()

        ax.text(*text_point, label, fontsize=12, ha="center", va="center", color=brightness(color, -0.4))
        ax.fill(*self.polygon.exterior.xy, alpha=0.5, color=color)


@dataclass
class Process:
    name: str
    application: Application
    mns: int
    max_delay_ms: int
    min_reliability: float
    aoi: AoI

    def __post_init__(self):
        self.aoi = AoI(**self.aoi)

        if isinstance(self.application, Application):
            return

        self.application = Application(**self.application)


@dataclass
class ProcessSplit(Process):
    split_name: str = None
    parent_process_name: str = None


@dataclass
class Host:
    label: str = None
    cpu_ghz: float = None
    ram_gb: float = None
    infinite_parallelism: bool = False
    qtime_dist_function: callable = None


@dataclass
class BR(Host):
    pos: sh.geometry.Point = None
    radius: float = None
    color: str = None
    coverage_area: sh.geometry.Polygon = None

    def __post_init__(self):
        if isinstance(self.pos, sh.geometry.Point):
            return

        self.pos = sh.geometry.Point(self.pos)


@dataclass
class CloudNode(Host):
    def __post_init__(self):
        self.infinite_parallelism = True


@dataclass
class Link:
    label: str
    delay_distribution: DistributionDescriptor

    def __post_init__(self):
        if isinstance(self.delay_distribution, DistributionDescriptor):
            return

        self.delay_distribution = DistributionDescriptor(**self.delay_distribution)
