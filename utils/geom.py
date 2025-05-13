import matplotlib.pyplot as plt
import math
import shapely.geometry as sh
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon


def generate_random_points_in_polygon(polygon, num_points, plot=False):
    """
    Generate random points within a polygon using Numpy and Geopandas.

    :param polygon: A Shapely Polygon object
    :param num_points: Number of random points to generate
    :param plot: Boolean, if True, plots the results
    :return: GeoDataFrame with points inside the polygon
    """
    # Step 1: Generate random points in the bounding box of the polygon
    minx, miny, maxx, maxy = polygon.bounds
    x = np.random.uniform(minx, maxx, num_points)
    y = np.random.uniform(miny, maxy, num_points)

    # Step 2: Create a GeoDataFrame with the random points
    df = pd.DataFrame()
    df["points"] = list(zip(x, y))  # Combine x and y into tuples
    df["points"] = df["points"].apply(Point)  # Convert tuples to Shapely Points
    gdf_points = gpd.GeoDataFrame(df, geometry="points")

    # Step 3: Create a GeoDataFrame for the polygon
    gdf_poly = gpd.GeoDataFrame(index=["polygon"], geometry=[polygon])

    # Step 4: Perform spatial join to filter points within the polygon
    points_in_poly = gpd.sjoin(gdf_points, gdf_poly, predicate="within", how="inner")

    # Step 5: Extract the list of points
    points_list = list(points_in_poly.geometry)

    return points_list


def sort_points_clockwise(points):
    """
    Sort points in clockwise order around their centroid.

    Parameters:
    points (list of tuples): List of (x, y) coordinates.

    Returns:
    list of tuples: Points sorted in clockwise order.
    """
    # Calculate centroid
    centroid_x = sum(p.x for p in points) / len(points)
    centroid_y = sum(p.y for p in points) / len(points)

    # Sort points by angle from centroid
    points.sort(
        key=lambda point: math.atan2(point.y - centroid_y, point.x - centroid_x)
    )
    return points


def extend_linestring(s, scale=1e6, one_direction=False):
    """
    Extend a LineString segment to a very large length.

    Parameters:
    s (LineString): Input segment as a LineString.
    scale (float): Factor to extend the segment.
    one_direction (bool): If True, extend only towards the end point (B).
                          If False, extend in both directions.

    Returns:
    LineString: The extended LineString.
    """
    # Ensure the input is a LineString
    if not isinstance(s, LineString):
        raise ValueError("Input must be a LineString.")

    # Extract the endpoints of the LineString
    a = Point(s.coords[0])
    b = Point(s.coords[-1])

    # Compute the direction vector
    direction = (b.x - a.x, b.y - a.y)

    # Normalize the direction vector
    length = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
    if length == 0:
        raise ValueError("The LineString's endpoints are identical, cannot extend.")

    unit_vector = (direction[0] / length, direction[1] / length)

    # Scale the vector to a large length
    extension = (unit_vector[0] * scale, unit_vector[1] * scale)

    if one_direction:
        # Extend only towards B
        end = Point(b.x + extension[0], b.y + extension[1])  # Extend forward
        return LineString([a, end])
    else:
        # Extend in both directions
        start = Point(a.x - extension[0], a.y - extension[1])  # Extend backward
        end = Point(b.x + extension[0], b.y + extension[1])  # Extend forward
        return LineString([start, end])


def compute_coverage_area(point, radius, obstacles, boundary):
    # extract the walls from the obstacles
    walls = [o.polygon for o in obstacles if o.is_wall]
    obstacles = [o.polygon for o in obstacles if not o.is_wall]

    # for each obstacle, convert its edges to walls
    for obstacle in obstacles:
        for i in range(len(obstacle.boundary.coords) - 1):
            walls.append(
                sh.LineString(
                    [obstacle.boundary.coords[i], obstacle.boundary.coords[i + 1]]
                )
            )

    # convert boundary to walls
    boundary: sh.Polygon = boundary.polygon

    boundary_walls = [
        LineString(
            [
                boundary.boundary.coords[i],
                boundary.boundary.coords[i + 1],
            ]
        )
        for i in range(len(boundary.boundary.coords) - 1)
    ]

    # create cutting polygons
    cutting_polygons = []
    for wall in walls:
        a, b = list(wall.boundary.geoms)

        # compute the projection of a and b on the boundary
        lines = [
            extend_linestring(LineString([point, a]), 1e6, one_direction=True),
            extend_linestring(LineString([point, b]), 1e6, one_direction=True),
        ]

        # find the intersection points of the pa and pb lines with any boundary and limit to boundary
        inters = []
        for boundary_wall in boundary_walls:
            for line in lines:
                intersections = boundary_wall.intersection(line)
                if not intersections.is_empty:
                    # compute the projection of the intersection point to solve numerical issues
                    projection = sh.Point(
                        extend_linestring(LineString([point, intersections]), 2).coords[
                            -1
                        ]
                    )
                    inters.append(projection)

        inters = [p for p in inters if not p.is_empty]

        # remove duplicates (e.g. when point is colinear with a wall)
        inters = list(set(inters))

        # create the cutting polygon using the intersection points (sort them to avoid self-intersections)
        poly = Polygon(sort_points_clockwise([a, b, *inters]))
        cutting_polygons.append(poly)

    # compute the region viewed by the point
    covered_region = point.buffer(radius).intersection(boundary)
    for obstacle in obstacles:
        covered_region = covered_region.difference(obstacle)

    for poly in cutting_polygons:
        covered_region = covered_region.difference(poly)

    # if the covered region is a multipolygon, select the polygon which contains the point
    if covered_region.geom_type == "MultiPolygon":
        for g in covered_region.geoms:
            if g.geom_type == "Polygon" and point.within(g):
                covered_region = g
                break

    return covered_region


def _compute_coverage_perimeter_old(
    rd_position: sh.Point, radius: float, obstacles_list
):

    # make sure rd_position is a Point, otherwise raise an error
    assert isinstance(rd_position, sh.Point)

    # utility function
    scale = lambda seg, val: sh.LineString(
        [
            seg.coords[0],
            sh.Point(
                seg.coords[0][0] + (seg.coords[1][0] - seg.coords[0][0]) * val,
                seg.coords[0][1] + (seg.coords[1][1] - seg.coords[0][1]) * val,
            ),
        ]
    )

    # divide obstacles in 2 lists: walls and other obstacles
    walls = [o.polygon for o in obstacles_list if o.is_wall]
    obstacles = [o.polygon for o in obstacles_list if not o.is_wall]

    # for each obstacle, convert its edges to walls
    for obstacle in obstacles:
        for i in range(len(obstacle.boundary.coords) - 1):
            walls.append(
                sh.LineString(
                    [obstacle.boundary.coords[i], obstacle.boundary.coords[i + 1]]
                )
            )

    circles = []
    for wall in walls:
        circle = rd_position.buffer(radius)

        # check if circle does not intersect with wall
        if not circle.intersects(wall):
            circles.append(circle)
            continue

        # count intersections on border
        interspoint = circle.exterior.intersection(wall)

        inters = None
        if isinstance(interspoint, sh.Point):
            inters = [interspoint]
        elif isinstance(interspoint, sh.MultiPoint):
            inters = [p for p in interspoint.geoms]
        else:
            inters = []

        segments = []
        points = []

        if len(inters) == 0:
            # create 2 segments with the points of the wall inside the circle
            for p in wall.coords:
                seg = sh.LineString([rd_position, sh.Point(p[0], p[1])])
                seg = scale(seg, 1000)
                segments.append(
                    sh.LineString(
                        [
                            (seg.coords[0][0], seg.coords[0][1]),
                            (seg.coords[1][0], seg.coords[1][1]),
                        ]
                    )
                )

                # points are the points of the wall inside the circle
                points.append(sh.Point(p[0], p[1]))

        if len(inters) == 1:
            # create 1 segment from (x,y) to intersection
            seg = sh.LineString(
                [rd_position, sh.Point(inters[0].coords[0][0], inters[0].coords[0][1])]
            )

            seg = scale(seg, 1000)
            segments.append(
                sh.LineString(
                    [
                        (seg.coords[0][0], seg.coords[0][1]),
                        (seg.coords[1][0], seg.coords[1][1]),
                    ]
                )
            )

            # 1st point is the intersection
            points.append(sh.Point(inters[0].coords[0][0], inters[0].coords[0][1]))

            # 2nd segment is from (x,y) to the point of the wall inside the circle
            for p in wall.coords:
                if sh.Point(p).within(circle):
                    seg = sh.LineString([rd_position, sh.Point(p[0], p[1])])
                    seg = scale(seg, 1000)
                    segments.append(
                        sh.LineString(
                            [
                                (seg.coords[0][0], seg.coords[0][1]),
                                (seg.coords[1][0], seg.coords[1][1]),
                            ]
                        )
                    )

                    # 2nd point is the point of the wall inside the circle
                    points.append(sh.Point(p[0], p[1]))

        if len(inters) == 2:
            # create 2 segments from (x,y) to each point of intersection
            for p in inters:
                seg = sh.LineString(
                    [rd_position, sh.Point(p.coords[0][0], p.coords[0][1])]
                )
                seg = scale(seg, 1000)
                segments.append(
                    sh.LineString(
                        [
                            (seg.coords[0][0], seg.coords[0][1]),
                            (seg.coords[1][0], seg.coords[1][1]),
                        ]
                    )
                )

            # points are the intersections with the wall
            points.append(sh.Point(inters[0].coords[0][0], inters[0].coords[0][1]))
            points.append(sh.Point(inters[1].coords[0][0], inters[1].coords[0][1]))

        assert len(segments) == 2
        assert len(points) == 2

        # create polygon from points
        polygon = sh.Polygon(
            [points[0], segments[0].coords[1], segments[1].coords[1], points[1]]
        )

        # cut circle with polygon
        new_circle = circle.difference(polygon)
        circles.append(new_circle)

    # intersect all circles
    circle = circles[0]
    for i in range(1, len(circles)):
        circle = circle.intersection(circles[i], grid_size=0.1)

        if (
            circle.geom_type == "GeometryCollection"
            or circle.geom_type == "MultiPolygon"
        ):
            # select polygon inside geometry collection
            for geom in circle.geoms:
                if geom.geom_type == "Polygon":
                    circle = geom
                    break

    # return the intersection
    return circle.exterior.coords
