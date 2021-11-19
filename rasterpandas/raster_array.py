from typing import List
from pandas.core.algorithms import isin

import xarray as xr
import pandas as pd
import numpy as np


@pd.api.extensions.register_extension_dtype
class RasterDtype(pd.api.extensions.ExtensionDtype):
    name : str = "raster"

    @property
    def type(self) -> type:
        return xr.DataArray
    
    @classmethod
    def construct_array_type() -> type:
        return RasterArray


class RasterArray(pd.api.extensions.ExtensionArray):
    # TODO: look at how rasterframes handles rasters.
    # Seems like they have a struct with "context" (extent, crs) and the data

    def __init__(self, items: List[xr.DataArray]):
        items = list(items)
        self._items = items
        self._dtype = RasterDtype()
    
    @property
    def dtype(self) -> RasterDtype:
        return self._dtype
        
    @classmethod
    def _from_sequence(cls, scalars: List[xr.DataArray], *, dtype=None, copy=False):
        # TODO: dtype, copy
        return cls(scalars)
    
    @classmethod
    def _from_factorized(cls, values, original):
        raise NotImplementedError
        
    def __getitem__(self, key):
        if np.isscalar(key):
            return self._items[key]
        else:
            return type(self)(self._items[key])
    
    def __len__(self):
        return len(self._items)
    
    def __eq__(self, other):
        if type(self) == type(other):
            return self._items  == other.items
        return NotImplemented

    def nbytes(self):
        return sum(item.nbytes for item in self._items)
    
    def isna(self):
        return np.zeros(len(self), dtype=False)
    
    def copy(self):
        return type(self)(self._items)

    @classmethod
    def _concat_same_type(cls, to_concat):
        items = []
        for arr in to_concat:
            items.extend(arr._items)
        return cls(items)

    def _formatter(self, boxed=False):
        return lambda arr: "<Raster>(arr)"

    #  ----------
    # helpers
    def apply(self, f, *args, **kwargs):
        # what kind of alignment do we expect?
        return type(self)([
            f(item, *args, **kwargs) for item in self._items
        ])

    def _format_array(self, *args, **kwargs):
        return [
            repr(item) for item in self._items
        ]


@pd.api.extensions.register_dataframe_accessor("raster")
class RasterAccessor:
    def __init__(self, pandas_obj):
        self._df = pandas_obj

    def ndvi(self, nir=None, red=None):
        import xrspatial

        # TODO: standard name handling; default to eo:common_name, string looks up from self
        nir = self._df[nir]
        red = self._df[red]
        nir, red = nir.align(red)
        # TODO: mask with NA
        # breakpoint()
        return pd.Series(
            RasterArray(
                xrspatial.ndvi(nir_agg, red_agg)
                for nir_agg, red_agg in zip(nir.tolist(), red.tolist())
            ),
            name="ndvi"
        )


class AssetGetter:
    def __getitem__(self, key):
        return ...
