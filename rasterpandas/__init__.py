"""rasterpandas"""
__version__ = "0.0.0"

from .raster_array import RasterArray, RasterDtype
from .stac_array import ItemArray, STACDtype

del raster_array
del stac_array

__all__ = [
    "RasterArray",
    "RasterDtype",
    "ItemArray",
    "STACDtype",
    "__version__",
]
