from typing import Optional
import numpy as np
from geocube.api.core import make_geocube
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import geopandas as gpd

from . import commons


def optimize_coverage(
    polygon: gpd.GeoDataFrame,
    n_points: int = 10,
    min_dist_edges: Optional[float] = None,
    resolution: float = 1,
    debug: bool = False,
) -> gpd.GeoDataFrame:
    """
    Create a representative sampling of a polygon using KMeans clustering.

    Args:
        polygon (gpd.GeoDataFrame): Polygon to sample.
        n_points (int): Number of sampling points. Defaults to 10.
        min_dist_edges (Optional[float]): Minimum distance to polygon edges. If None, no minimum distance is enforced.
        It has the same unit as the input data. Defaults to None.
        resolution (float): Resolution of the rasterized polygon used for clustering. It is in the same unit of the input data. Defaults to 1.
        debug (bool): If True, return the cluster labels as a DataArray together with the sampling points. Defaults to False.

    Returns:
        gpd.GeoDataFrame: Sampling points.
    """
    commons.validate_single_polygon(polygon)
    assert (
        min_dist_edges is None or min_dist_edges >= 0
    ), "min_dist_edges must be None or a positive value"
    # Avoid modifying the input polygon
    polygon = polygon.copy()
    # Multipolygon is converted to polygon if needed by considering only the largest polygon
    polygon = polygon.map(commons.multi_polygon_to_polygon)
    # Prepare input data
    polygon = polygon[["geometry"]]
    polygon["auxiliary_col"] = 1
    # Apply buffer
    if min_dist_edges is not None:
        polygon["geometry"] = polygon["geometry"].buffer(-min_dist_edges)
    # Rasterize
    polygon_raster = make_geocube(vector_data=polygon, resolution=(resolution, -resolution))
    polygon_raster = polygon_raster["auxiliary_col"]
    # Convert DataArray to DataFrame
    polygon_df = polygon_raster.drop_vars("spatial_ref").to_dataframe().reset_index(drop=False)
    nodata_idx = polygon_df["auxiliary_col"].isna()
    polygon_df_filtered = polygon_df[~nodata_idx]
    # Run KMeans to cluster pixels
    features = polygon_df_filtered[["y", "x"]].values
    features = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=n_points, random_state=0).fit(features)
    # For each centroid, get the closest point, and create a GeoDataFrame with the sampling points
    kmeans_centroids = kmeans.cluster_centers_
    sampling_pixels_idx = euclidean_distances(kmeans_centroids, features).argmin(axis=1)
    sampling_pixels_coords = polygon_df_filtered.iloc[sampling_pixels_idx][["x", "y"]].values
    # Create a GeoDataFrame with the sampling points
    sampling_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(sampling_pixels_coords[:, 0], sampling_pixels_coords[:, 1]),
        crs=polygon_raster.rio.crs,
    )
    if debug:
        # Assign cluster labels to the original dataframe
        cluster_labels = kmeans.labels_
        polygon_df["cluster"] = np.nan
        polygon_df.loc[~nodata_idx, "cluster"] = cluster_labels
        # Convert DataFrame back to DataArray
        polygon_clusters_raster = polygon_df.set_index(["y", "x"]).to_xarray()["cluster"]
        polygon_clusters_raster = polygon_clusters_raster.rio.write_crs(polygon_raster.rio.crs)
        return sampling_points, polygon_clusters_raster

    return sampling_points


def random_sampling(
    polygon: gpd.GeoDataFrame, n_points: int = 10, min_dist_edges: Optional[float] = None
) -> gpd.GeoDataFrame:
    """
    Create a random sampling within a polygon.
    This function is a wrapper around the `sample_points` method from the `geopandas` library.

    Args:
        polygon (gpd.GeoDataFrame): Polygon to sample.
        n_points (int): Number of sampling points. Defaults to 10.
        min_dist_edges (Optional[float]): minimum distance to polygon edges. If None, no minimum distance is enforced.
        It has the same unit as the input data. Defaults to None.

    Returns:
        gpd.GeoDataFrame: Sampling points.
    """
    commons.validate_single_polygon(polygon)
    assert (
        min_dist_edges is None or min_dist_edges >= 0
    ), "min_dist_edges must be None or a positive value"
    # Avoid modifying the input polygon
    polygon = polygon.copy()
    # Multipolygon is converted to polygon if needed by considering only the largest polygon
    polygon = polygon.map(commons.multi_polygon_to_polygon)
    # Apply buffer
    if min_dist_edges is not None:
        polygon["geometry"] = polygon["geometry"].buffer(-min_dist_edges)
    return polygon.sample_points(size=n_points, method="uniform", rng=0)
