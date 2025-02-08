import numpy as np
from geocube.api.core import make_geocube
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import geopandas as gpd


def optimize_coverage(
    polygon: gpd.GeoDataFrame, n_points: int = 10, resolution: float = 5, debug: bool = False
) -> gpd.GeoDataFrame:
    """
    Create a representative sampling of a polygon using KMeans clustering.

    Args:
        polygon (gpd.GeoDataFrame): Polygon to sample.
        n_points (int): Number of sampling points.
        resolution (float): Resolution of the rasterized polygon used for clustering.
        debug (bool): If True, return the cluster labels as a DataArray together with the sampling points.

    Returns:
        gpd.GeoDataFrame: Sampling points.
    """
    assert len(polygon) == 1, "Only one polygon is allowed"
    assert polygon.crs.is_projected, "CRS must be projected"
    # Prepare input data
    polygon = polygon[["geometry"]]
    polygon["auxiliary_col"] = 1
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
