from shapely import Geometry
import geopandas as gpd


def multi_polygon_to_polygon(geom: Geometry) -> Geometry:
    """Convert a MultiPolygon to a Polygon by taking the largest polygon.
    Any other type of geometry is returned as is.
    """
    assert isinstance(geom, Geometry), "Input must be a Shapely geometry"
    if geom.geom_type == "MultiPolygon":
        largest_polygon = max(geom.geoms, key=lambda p: p.area)
        return largest_polygon
    return geom


def validate_single_polygon(polygon: gpd.GeoSeries) -> None:
    """Validate that the input is a single polygon.

    Args:
        polygon (gpd.GeoSeries): Polygon to validate.

    Raises:
        AssertionError: If the input is not a single polygon.
    """
    assert len(polygon) == 1, "Only one polygon is allowed"
    assert polygon.crs.is_projected, "CRS must be projected"
