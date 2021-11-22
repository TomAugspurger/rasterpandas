from typing import List
import operator
import weakref

import pandas as pd
import numpy as np
import pystac
import stackstac


@pd.api.extensions.register_extension_dtype
class STACDtype(pd.api.extensions.ExtensionDtype):
    name = "stac"

    @property
    def type(self):
        return pystac.Item
    
    @classmethod
    def construct_array_type():
        return ItemArray



class AssetGetter:
    def __init__(self, array):
        self._array = weakref.ref(array)

    def __getitem__(self, key):
        # TODO: return an AssetArray.
        if isinstance(key, str):
            # a single one
            return [item.assets[key] for item in self._array()]
        else:
            arrays = []
            for k in key:
                arrays.append([item.assets[k] for item in self.array()])
            return arrays


class ItemArray(pd.api.extensions.ExtensionArray):

    def __init__(self, items, clone_items=True):
        items = pystac.ItemCollection(items, clone_items=True)
        self._data = items
        self._dtype = STACDtype()
        # self.assets = AssetGetter(self)
    
    @property
    def dtype(self):
        return self._dtype
        
    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        # TODO: dtype
        return cls(scalars, clone_items=copy)
    
    @classmethod
    def _from_factorized(values, original):
        raise NotImplementedError
        
    def __getitem__(self, key):
        if np.isscalar(key):
            return self._data[key]
        else:
            return type(self)(np.asarray(self._data, dtype=object)[key])
    
    def __len__(self):
        return len(self._data)

    def __eq__(self, other):
        if type(self) == type(other):
            return self._data  == other.items
        return NotImplemented

    def nbytes(self):
        return 8  # TODO: use to_dict()?
    
    def isna(self):
        return np.zeros(len(self), dtype=bool)
    
    def copy(self):
        return type(self)(self._data)

    @classmethod
    def _concat_same_type(cls, to_concat):
        items = []
        for arr in to_concat:
            items.extend(arr._data.items)
        return cls(items)

    # -----------------
    # sugar
    def _apply(self, f, dtype=object):
        return [f(x) for x in self._data]
    
    @property
    def id(self):
        return pd.array(self._apply(operator.attrgetter("id")))

    @property
    def bbox(self):
        return self._apply(operator.attrgetter("bbox"))

    @property
    def collection_id(self):
        return pd.array(self._apply(operator.attrgetter("collection_id")))

    @property
    def datetime(self):
        return pd.to_datetime(self._apply(operator.attrgetter("datetime"))).array

    @property
    def stac_extensions(self):
        return pd.array(self._apply(operator.attrgetter("stac_extensions")))

    @property
    def assets(self):
        # TODO: ListArray
        pass


@pd.api.extensions.register_series_accessor("stac")
# @pd.api.extensions.register_dataframe_accessor("stac")
class STACAccessor:
    def __init__(self, pandas_obj):
        self._series = pandas_obj

    @property
    def asset_names(self):
        # Are we assuming assets are uniform? Taking the union? ...
        return list(self._series.array._data[0].assets)

    @property
    def datetime(self):
        return pd.Series(self._series.array.datetime, index=self._series.index, name="datetime")

    def assets(self):
        return pd.Series(self._series.array.assets, index=self._series.index, name="assets")

    def with_rasters(self, assets=None):
        from .raster_array import RasterArray

        series = []
        assets = assets or self.asset_names

        for asset in assets:
            ra = RasterArray([
                stackstac.stack(item, assets=[asset], chunksize=-1).squeeze() for item in self._series.array._data
            ])
            series.append(pd.Series(ra, index=self._series.index, name=asset))

        df = pd.concat(series, axis="columns")
        return pd.concat([self._series, df], axis="columns")


@pd.api.extensions.register_extension_dtype
class AssetDtype(pd.api.extensions.ExtensionDtype):
    name = "asset"

    @property
    def type(self):
        return pystac.Asset
    
    @classmethod
    def construct_array_type():
        return AssetArray



class AssetArray(pd.api.extensions.ExtensionArray):
    def __init__(self, assets: List[pystac.Asset]):
        self._data = list(assets)
        self._dtype = AssetDtype()

    @property
    def dtype(self):
        return self._dtype
        
    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        # TODO: dtype, copy
        return cls(scalars)
    
    @classmethod
    def _from_factorized(values, original):
        raise NotImplementedError
        
    def __getitem__(self, key):
        if np.isscalar(key):
            return self._data[key]
        elif isinstance(key, slice):
            return type(self)(self._data[key])
        else:
            return type(self)([self._data[k] for k in key])

    def __len__(self):
        return len(self._data)

    def __eq__(self, other):
        if type(self) == type(other):
            return self._data  == other._data
        return NotImplemented

    def nbytes(self):
        return 8  # TODO: use to_dict()?
    
    def isna(self):
        return np.zeros(len(self), dtype=bool)
    
    def copy(self):
        return type(self)(self._data)

    @classmethod
    def _concat_same_type(cls, to_concat):
        items = []
        for arr in to_concat:
            items.extend(arr._data.items)
        return cls(items)

