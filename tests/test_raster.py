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
        xr.DataArray(values, coords=dict(x=x, y=y), dims=("y", "x")) for _ in range(100)
    ]
    return rasterpandas.RasterArray(arrays)


@pytest.fixture
def na_value():
    return pd.NA


class TestConstructors(base.BaseConstructorsTests):
    @pytest.mark.xfail(reason="upstream -- length", strict=True)
    def test_series_constructor_scalar_with_index(self, data, dtype):
        return super().test_series_constructor_scalar_with_index(data, dtype)


class TestCasting(base.BaseCastingTests):
    pass


class TestDim2CompatTests(base.Dim2CompatTests):
    pass


class TestDtype(base.BaseDtypeTests):
    pass

class TestGetitem(base.BaseGetitemTests):
    pass


class TestGroupby(base.BaseGroupbyTests):
    pass


class TestInterface(base.BaseInterfaceTests): pass
class TestParsing(base.BaseParsingTests): pass
class TestMethods(base.BaseMethodsTests): pass
class TestMissing(base.BaseMissingTests): pass
class TestArithmeticOps(base.BaseArithmeticOpsTests):
    pass

class TestComparision(base.BaseComparisonOpsTests): pass
class TestOps(base.BaseOpsUtil): pass
class TestUnaryOps(base.BaseUnaryOpsTests): pass
class TestPrinting(base.BasePrintingTests): pass
class TestBooleanReduce(base.BaseBooleanReduceTests): pass
class TestNoBooleanReduce(base.BaseNoReduceTests): pass
class TestNumericBooleanReduce(base.BaseNumericReduceTests): pass
class TestReshaping(base.BaseReshapingTests): pass
class TestSetitem(base.BaseSetitemTests): pass
