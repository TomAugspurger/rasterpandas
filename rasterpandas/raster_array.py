from __future__ import annotations

from typing import List, Callable
from pandas.core.algorithms import isin

import xarray as xr
import xrspatial
import pandas as pd
import numpy as np


@pd.api.extensions.register_extension_dtype
class RasterDtype(pd.api.extensions.ExtensionDtype):
    name: str = "raster"

    @property
    def type(self) -> type:
        return xr.DataArray

    @classmethod
    def construct_array_type(cls) -> type:
        return RasterArray

    @property
    def na_value(self):
        return pd.NA


class RasterArray(pd.api.extensions.ExtensionArray):
    """
    A pandas ExtensionArray for storing raster imagery.

    Each element of the array is a 2-d DataArray.
    """

    # TODO: look at how rasterframes handles rasters.
    # Seems like they have a struct with "context" (extent, crs) and the data

    def __init__(self, items: List[xr.DataArray | pd.NA]):
        items = list(items)
        self._data = items
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
            return self._data[key]
        elif isinstance(key, slice):
            return type(self)(self._data[key])
        else:
            return type(self)([self._data[k] for k in key])

    def take(self, indices, allow_fill=False, fill_value=None):
        if np.max(indices) > len(self):
            raise IndexError()
        elif allow_fill and np.min(indices) < -1:
            raise ValueError()

        values = []
        for idx in indices:
            if allow_fill and idx == -1:
                values.append(fill_value)
            else:
                values.append(self[idx])
        return type(self)(values)

    def __len__(self):
        return len(self._data)

    def __eq__(self, other):
        if type(self) == type(other):
            return self._data == other.items
        return NotImplemented

    def nbytes(self):
        return sum(item.nbytes for item in self._data)

    def isna(self):
        return pd.array([x is pd.NA for x in self._data], dtype=pd.BooleanDtype())

    def copy(self):
        return type(self)(self._data)

    @classmethod
    def _concat_same_type(cls, to_concat):
        items = []
        for arr in to_concat:
            items.extend(arr._data)
        return cls(items)

    def _formatter(self, boxed=False):
        return lambda arr: "<Raster>(arr)"

    #  ----------
    # helpers
    def _apply(self, f, *args, **kwargs):
        # what kind of alignment do we expect?
        return type(self)([f(item, *args, **kwargs) for item in self._data])

    def _format_array(self, *args, **kwargs):
        return [repr(item) for item in self._data]


@pd.api.extensions.register_dataframe_accessor("raster")
class DataFrameRasterAccessor:
    def __init__(self, pandas_obj):
        self._df = pandas_obj

    def ndvi(self, nir=None, red=None):
        # TODO: standard name handling; default to eo:common_name, string looks up from self
        nir = self._df[nir]
        red = self._df[red]
        nir, red = nir.align(red)
        return pd.Series(
            RasterArray(
                xrspatial.ndvi(nir_agg, red_agg)
                for nir_agg, red_agg in zip(nir.tolist(), red.tolist())
            ),
            name="ndvi",
        )


@pd.api.extensions.register_series_accessor("raster")
class SeriesRasterAccessor:
    def __init__(self, pandas_obj):
        self._series = pandas_obj

    def apply(self, f: Callable, *args, **kwargs):
        name = f.__name__

        result = pd.Series(
            RasterArray([
                f(item) for item in self._series.tolist()
            ]),
        index=self._series.index, name=name)
        return result


class AssetGetter:
    def __getitem__(self, key):
        return ...
