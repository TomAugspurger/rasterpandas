import numpy as np
import xarray as xr
import pandas as pd
from pandas.tests.extension import base
import pytest

import rasterpandas


values = np.empty(shape=(10, 10))


@pytest.fixture
def dtype():
    return rasterpandas.RasterDtype()


@pytest.fixture
def data():
    x = xr.DataArray(range(10), name="x", dims="x")
    y = xr.DataArray(range(10), name="y", dims="y")
    arrays = [
        xr.DataArray(values, coords=dict(x=x, y=y), dims=("y", "x"))
        for _ in range(100)
    ]
    return rasterpandas.RasterArray(arrays)


@pytest.fixture
def na_value():
    return pd.NA


class TestConstructors(base.BaseConstructorsTests):
    @pytest.mark.xfail(reason="upstream -- length", strict=True)
    def test_series_constructor_scalar_with_index(self, data, dtype):
        return super().test_series_constructor_scalar_with_index(data, dtype)
