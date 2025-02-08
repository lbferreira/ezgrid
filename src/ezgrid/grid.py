from typing import Optional

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Point


def find_polygon_angle(polygon: gpd.GeoSeries) -> float:
    # Find the largest distance between vertices
    if len(polygon) > 1:
        raise ValueError("The input GeoSeries should contain only one polygon")
    rotated_rect = polygon.minimum_rotated_rectangle()
    coords = rotated_rect.iloc[0].exterior.coords.xy
    coords_x = coords[0]
    coords_y = coords[1]
    x_ini = coords_x[0]
    y_ini = coords_y[0]
    max_dist = 0
    angle = None
    for x_end, y_end in zip(coords_x[1:], coords_y[1:]):
        dist = ((x_end - x_ini) ** 2 + (y_end - y_ini) ** 2) ** 0.5
        if dist > max_dist:
            max_dist = dist
            # With two vertices that define a line, compute the angle of the line with respect to the y-axis
            # angle = np.arctan2(y_end-y_ini, x_end-x_ini)
            angle = np.arctan2(x_end - x_ini, y_end - y_ini)
            # convert to degrees
            angle = np.degrees(angle)
        x_ini = x_end
        y_ini = y_end
    if angle is None:
        raise Exception("Something went wrong when computing the angle of the polygon.")
    return angle


def create_grid(polygon: gpd.GeoSeries, points_dist: float) -> gpd.GeoDataFrame:
    xmin, ymin, xmax, ymax = polygon.total_bounds
    x_coords = np.arange(xmin, xmax, points_dist)
    y_coords = np.arange(ymin, ymax, points_dist)
    # Convert to Point objects
    points = [Point(x, y) for x in x_coords for y in y_coords]
    # To GeoDataFrame
    return gpd.GeoDataFrame(geometry=points, crs=polygon.crs)


def _error_function(
    x_offset,
    y_offset,
    x_coords,
    y_coords,
    points_dist,
    target_polygon: shapely.geometry.Polygon,
):
    x_coords_iter = x_coords + x_offset
    y_coords_iter = y_coords + y_offset
    # To GeoDataFrame
    grid_iter = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(x_coords_iter, y_coords_iter)]
    )
    # Crop the grid to keep only the points within the polygon
    grid_iter = grid_iter[grid_iter.intersects(target_polygon)].reset_index(drop=True)

    # Distance of each point to the polygon
    xmin, ymin, xmax, ymax = grid_iter.total_bounds
    dists = grid_iter.distance(target_polygon.exterior).values
    dists[dists > 1.5 * points_dist] = 0

    nb_points = len(grid_iter)
    grid_x_center = (xmin + xmax) / 2
    grid_y_center = (ymin + ymax) / 2
    xmin, ymin, xmax, ymax = target_polygon.bounds
    target_polygon_x_center = (xmin + xmax) / 2
    target_polygon_y_center = (ymin + ymax) / 2
    centers_dist = (
        (grid_x_center - target_polygon_x_center) ** 2
        + (grid_y_center - target_polygon_y_center) ** 2
    ) ** 0.5

    if nb_points == 0:
        return np.inf
    else:
        # error = (((dists)**0.5).sum() / nb_polygons) + ((centers_dist)**2)
        error = -((dists**0.5).sum() * nb_points) + ((centers_dist) ** 2)
    return error


def optimize_grid(
    grid: gpd.GeoDataFrame,
    points_dist: float,
    polygon: gpd.GeoSeries,
    buffer: Optional[float] = None,
    max_offset: Optional[float] = None,
    n_trials=200,
) -> gpd.GeoDataFrame:
    assert len(polygon) == 1, "The input GeoSeries should contain only one polygon"
    assert grid.crs == polygon.crs, "The grid and the polygon should have the same CRS"
    buffer = -points_dist / 4 if buffer is None else buffer
    max_offset = 0.75 * points_dist if max_offset is None else max_offset

    # TODO: SImplify is hard-coded to 1, should be a parameter

    polygon_geom = polygon.buffer(buffer).iloc[0]
    x_coords = grid.geometry.x.values
    y_coords = grid.geometry.y.values

    steps = int(n_trials**0.5)
    x_offset_candidates = np.linspace(-max_offset, max_offset, steps)
    y_offset_candidates = np.linspace(-max_offset, max_offset, steps)
    min_error = np.inf
    best_x_offset = None
    best_y_offset = None
    for x_offset in x_offset_candidates:
        for y_offset in y_offset_candidates:
            error = _error_function(
                x_offset,
                y_offset,
                x_coords,
                y_coords,
                points_dist,
                polygon_geom.simplify(1),
            )
            if error < min_error:
                min_error = error
                best_x_offset = x_offset
                best_y_offset = y_offset

    x_coords_opt = x_coords + best_x_offset
    y_coords_opt = y_coords + best_y_offset
    # Recreate the grid as a GeoDataFrame with the optimized points
    grid_opt = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(x_coords_opt, y_coords_opt)], crs=grid.crs
    )
    # Crop the grid to keep only the points within/intersects the polygon
    grid_opt = grid_opt[grid_opt.intersects(polygon_geom)].reset_index(drop=True)
    return grid_opt


def revert_rotation(
    grid_rotated: gpd.GeoDataFrame, angle: float, polygon_rotated: gpd.GeoSeries
) -> gpd.GeoDataFrame:
    # pol_rot_center_x = polygon_rotated.centroid.x[0]
    # pol_rot_center_y = polygon_rotated.centroid.y[0]
    xmin, ymin, xmax, ymax = polygon_rotated.total_bounds
    pol_rot_center_x = (xmin + xmax) / 2
    pol_rot_center_y = (ymin + ymax) / 2
    grid_reverted = grid_rotated.copy()
    grid_reverted["geometry"] = grid_reverted.rotate(
        -angle, origin=(pol_rot_center_x, pol_rot_center_y)
    )
    return grid_reverted


def create_optimized_grid(
    polygon: gpd.GeoSeries,
    points_dist: float,
    buffer: Optional[float] = None,
    n_trials: int = 200,
    max_offset: Optional[float] = None,
) -> gpd.GeoDataFrame:
    angle = find_polygon_angle(polygon)
    polygon_rotated = polygon.rotate(angle)
    grid_rotated = create_grid(polygon_rotated, points_dist=points_dist)
    grid_rotated_opt = optimize_grid(
        grid_rotated, points_dist, polygon_rotated, buffer, max_offset, n_trials
    )
    grid_opt = revert_rotation(grid_rotated_opt, angle, polygon)
    return grid_opt
