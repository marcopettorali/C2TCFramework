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

import shapely as sh


class Obstacle:
    def __init__(self, points, label: str = None):
        self.points = points
        self.label = label

        if len(self.points) == 2:
            self.polygon = sh.LineString(self.points)
            self.is_wall = True
        else:
            self.polygon = sh.Polygon(points)
            self.is_wall = False


class WallObstacle(Obstacle):

    def __init__(self, start: sh.Point, end: sh.Point, label: str = None):
        points = [start, end]
        super().__init__(points, label)


class RectangularObstacle(Obstacle):

    def __init__(self, start: sh.Point, end: sh.Point, label: str = None):

        start = sh.Point(start)
        end = sh.Point(end)

        # if start and end are tuples convert them to Points
        p1 = sh.Point(min(start.x, end.x), min(start.y, end.y))
        p2 = sh.Point(max(start.x, end.x), max(start.y, end.y))
        points = [p1, sh.Point(p1.x, p2.y), p2, sh.Point(p2.x, p1.y)]
        super().__init__(points, label)


class SquareObstacle(RectangularObstacle):

    def __init__(self, center: sh.Point, side_length: float, label: str = None):

        center = sh.Point(center)

        p1 = sh.Point(center.x - side_length / 2, center.y - side_length / 2)
        p2 = sh.Point(center.x + side_length / 2, center.y + side_length / 2)

        super().__init__(p1, p2, label)


class CircularObstacle(Obstacle):

    def __init__(self, center: sh.Point, radius: float, label: str = None):
        points = sh.Point(center).buffer(radius).exterior.coords
        super().__init__(points, label)
