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
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import scipy.stats
import shapely as sh

from networking.entities import *
from networking.obstacle import *
from utils.geom import compute_coverage_area
from utils.plotting import bold


class Environment:
    _enable_multiprocessing = True

    def enable_multiprocessing():
        Environment._enable_multiprocessing = True
        print("Environment: multiprocessing enabled")

    def disable_multiprocessing():
        Environment._enable_multiprocessing = False
        print("Environment: multiprocessing disabled")

    def __init__(self, **kwargs):
        self.env_filename = kwargs.get("env_filename", None)
        self.width = kwargs.get("width", None)
        self.height = kwargs.get("height", None)
        self.obstacles = kwargs.get("obstacles", [])
        self.deployment = kwargs.get("deployment", [])

        # install all other kwargs as attributes
        for key, value in kwargs.items():
            if key not in ["width", "height", "obstacles", "deployment"]:
                setattr(self, key, value)

    def __repr__(self):
        return f"Environment(width={self.width}, height={self.height}, #obstacles={len(self.obstacles)}, #brs={len(self.deployment)})"

    __str__ = __repr__

    @staticmethod
    def load_env_from_file(env_filename):
        # load json file
        with open(env_filename, "r") as f:
            data = json.load(f)

        # load the width and height
        width = data["width"]
        height = data["height"]

        # load the obstacles
        obstacles = [Obstacle(obstacle) for obstacle in data["obstacles"]]

        return Environment(width=width, height=height, obstacles=obstacles, deployment=[], env_filename=env_filename)

    def load_deployment_from_file(self, deployment_filename, key=None):

        self.deployment_filename = deployment_filename

        # load json file
        with open(deployment_filename, "r") as f:
            data = json.load(f)

            # handle the key if it is provided
            if key is not None:
                data = data[key]

        self.load_deployment(data)

    def load_deployment(self, deployment):
        self.deployment = [BR(label=f"BR{i}", color="black", **br) for i, br in enumerate(deployment)]
        self.recompute_coverage_areas()

    def recompute_coverage_areas(self):
        # compute the coverage perimeter for each BR (use multiprocessing)
        if self._enable_multiprocessing:
            with mp.Pool(mp.cpu_count()) as pool:
                coverages = pool.starmap(
                    compute_coverage_area,
                    [
                        (
                            br.pos,
                            br.radius,
                            self.obstacles,
                            self.get_boundary_obstacle(),
                        )
                        for br in self.deployment
                    ],
                )
        else:
            coverages = [compute_coverage_area(br.pos, br.radius, self.obstacles, self.get_boundary_obstacle()) for br in self.deployment]
        for i, br in enumerate(self.deployment):
            br.coverage_area = sh.Polygon(coverages[i])

    def print(self, filepath):
        fig, ax = plt.subplots()
        self.draw(ax)
        fig.tight_layout()
        fig.savefig(filepath, bbox_inches="tight")

    def draw(self, ax: plt.Axes):
        # divide the area into polys and walls
        polys = []
        walls = []

        for obstacle in self.obstacles:
            if len(obstacle.points) == 2:
                walls.append(sh.LineString(obstacle.points))
            else:
                polys.append(sh.Polygon(obstacle.points))

        # Draw obstacles
        for obstacle in polys:
            x, y = obstacle.exterior.xy
            ax.fill(x, y, color="black")

        # Draw walls
        for wall in walls:
            x, y = wall.xy
            ax.plot(x, y, color="black")

        # draw the deployment
        for br in self.deployment:
            # plot the BR
            ax.plot(br.pos.x, br.pos.y, br.color, marker="s")

            # plot the coverage perimeter
            x, y = br.coverage_area.exterior.xy
            if br.color == "black":
                ax.plot(x, y, color="grey", linewidth=0.7, linestyle="dashed")
            else:
                # fill the coverage area
                ax.fill(x, y, color=br.color, alpha=0.3)

            # plot the BR name bottom right
            # if "labeloffset" in br:
            #     ax.text(br["pos"].x + br["labeloffset"][0], br["pos"].y + br["labeloffset"][1], br["label"], fontsize=8, horizontalalignment="center", verticalalignment="top")
            # else:
            ax.text(
                br.pos.x,
                br.pos.y,
                bold(br.label),
                fontsize=8,
                horizontalalignment="center",
                verticalalignment="center",
                color="white",
            )

        # draw the area
        area = [
            (0, 0),
            (0, self.height),
            (self.width, self.height),
            (self.width, 0),
        ]

        # draw borders of the area
        x, y = sh.Polygon(area).exterior.xy
        ax.plot(x, y, color="black")

        ax.set_aspect("equal", adjustable="box")

    def get_boundary_obstacle(self):
        return Obstacle(
            [(0, 0), (0, self.height), (self.width, self.height), (self.width, 0)],
            "boundary",
        )

    # check if a point is in an obstacle
    def is_in_obstacle(self, point: sh.Point):
        for obstacle in self.obstacles:
            if obstacle.polygon.contains(point):
                return True
        return False

    # check if a line is crossing an obstacle or if one end is in an obstacle
    def is_crossing_obstacle(self, line: sh.LineString):
        endpoints = [sh.geometry.Point(x) for x in [line.coords[0], line.coords[-1]]]
        for obstacle in self.obstacles:
            if obstacle.polygon.crosses(line) or any(obstacle.polygon.contains(point) for point in endpoints):
                return True
        return False

    def _check_point_job(self, x, y, area=None):

        point = sh.Point(x, y)

        # check if the point is in an obstacle
        if self.is_in_obstacle(point):
            return None

        # check if the point is outside the environment
        if area is not None and not (area.contains(point)):
            return None

        return point

    # generate a random point in the environment, not in an obstacle
    def random_point_uniform(self, npoints=1, area=None):
        # use multiprocessing to generate multiple points
        ret = []

        if area is None:
            xloc, xscale = 0, self.width
            yloc, yscale = 0, self.height
        else:
            xloc, xscale = area.bounds[0], area.bounds[2] - area.bounds[0]
            yloc, yscale = area.bounds[1], area.bounds[3] - area.bounds[1]

        while len(ret) < npoints:
            xs = scipy.stats.uniform(loc=xloc, scale=xscale).rvs(size=npoints - len(ret))
            ys = scipy.stats.uniform(loc=yloc, scale=yscale).rvs(size=npoints - len(ret))

            if self._enable_multiprocessing:
                # raise NotImplementedError("Multiprocessing is not supported for random_point_uniform")
                with mp.Pool(mp.cpu_count()) as pool:
                    points = pool.starmap(self._check_point_job, [(x, y, area) for x, y in zip(xs, ys)])
            else:
                points = [self._check_point_job(x, y, area) for x, y in zip(xs, ys)]

            ret.extend([p for p in points if p is not None])

        # if only one point is requested, return the point, otherwise return a list
        if npoints == 1:
            return ret[0]
        return ret

    # generate a random area in the environment, not intersecting obstacles
    def random_area_points(self, npoints: int, rand_x_gen: callable, rand_y_gen: callable):
        points = []
        for p in range(npoints):
            while True:
                candidate = self.random_point(rand_x_gen, rand_y_gen, area=self.get_boundary_obstacle().polygon)

                if len(points) > 0:
                    new_segments = [sh.LineString([points[-1], candidate])]

                    # if last point, two segments are created
                    if p == npoints - 1:
                        new_segments.append(sh.LineString([candidate, points[0]]))

                    # check if segment with previous point intersects obstacles
                    if any([self.is_crossing_obstacle(ns) for ns in new_segments]):
                        continue

                    # check if segment auto-intersects
                    if any([sh.LineString([points[i], points[i + 1]]).crosses(ns) for ns in new_segments for i in range(len(points) - 1)]):
                        continue

                # append the point
                points.append(candidate)
                break

        return points

    def random_trajectory_segment(self, start_point, n_points=1, area=None):
        points = [start_point]
        for _ in range(n_points):
            while True:
                candidate = self.random_point_uniform(1, area)
                if len(points) > 0:
                    new_segment = sh.LineString([points[-1], candidate])
                    if self.is_crossing_obstacle(new_segment):
                        continue
                points.append(candidate)
                break
        return points

    def get_walkable_polygon(self):
        # if there are no obstacles, return the whole area
        if len(self.obstacles) == 0:
            return self.width * self.height

        # if there are obstacles, cut the area
        area = sh.Polygon([(0, 0), (0, self.height), (self.width, self.height), (self.width, 0)])
        for obstacle in self.obstacles:
            area = area.difference(obstacle.polygon)
        return area


if __name__ == "__main__":
    env = Environment.load_env_from_file("environments/env1.json")
    env.load_deployment_from_file("deployments/env1.json", "hom")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    env.draw(ax)
    fig.tight_layout()

    plt.show()
