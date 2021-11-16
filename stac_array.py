import operator
import weakref

import pandas as pd
import numpy as np
import pystac

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
        self._items = items
        self._dtype = STACDtype()
        self.assets = AssetGetter(self)
    
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
            return self._items[key]
        else:
            return type(self)(np.asarray(self._items, dtype=object)[key])
    
    def __len__(self):
        return len(self._items)

    def __eq__(self, other):
        if type(self) == type(other):
            return self._items  == other.items
        return NotImplemented

    def nbytes(self):
        return 8  # TODO: use to_dict()?
    
    def isna(self):
        return np.zeros(len(self), dtype=False)
    
    def copy(self):
        return type(self)(self._items)

    @classmethod
    def _concat_same_type(cls, to_concat):
        items = []
        for arr in to_concat:
            items.extend(arr._items.items)
        return cls(items)

    # -----------------
    # sugar
    def _apply(self, f, dtype=object):
        return [f(x) for x in self._items]
    
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
        return pd.array(self._apply(operator.attrgetter("datetime")), dtype="datetime64[ns]")

    @property
    def stac_extensions(self):
        return pd.array(self._apply(operator.attrgetter("stac_extensions")))