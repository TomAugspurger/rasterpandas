import pystac
import numpy as np
import rasterio

def stacify():
    pass


def stacify_rasterio_dataset(ds: rasterio.DatasetReader) -> pystac.Item:
    pass


def stacify_xarray_datarray(ds) -> pystac.Item:
    ...


def stacify_url(str) -> pystac.Item:
    ...
    


if __name__ == "__main__":
    import pystac_client
    import geopandas
    import planetary_computer
    import pystac
    import pandas as pd

    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    items = catalog.search(
        collections=["landsat-8-c2-l2"],
        datetime="2021-07-01T08:00:00Z/2021-07-01T10:00:00Z"
    ).get_all_items()


    items = [planetary_computer.sign(item) for item in items]
    items = pystac.ItemCollection(items, clone_items=False)
    df = geopandas.GeoDataFrame.from_features(items.to_dict(), crs="epsg:4326")

    # https://github.com/geopandas/geopandas/issues/1208
    df["id"] = [x.id for x in items]
    m = df[["geometry", "id", "datetime"]].explore()