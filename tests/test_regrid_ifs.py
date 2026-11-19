"""
Tests for `anna_inference.regrid_ifs`: the smoothing, the selection of the
forecast and the regridding of a synthetic IFS forecast.

    uv run pytest tests
"""

import datetime

import numpy as np
import pytest
import xarray as xr

from anna_inference import regrid_ifs as ri
from anna_inference.regrid_common import BOUNDARY_LATS, BOUNDARY_LONS, LEVELS

ANALYSIS_TIME = datetime.datetime(2026, 10, 1, 6)
DURATION = datetime.timedelta(hours=18)
CYCLE = np.datetime64("2026-10-01T00", "ns")
# more levels than ANNA uses (all of them, and 500 hPa)
IFS_LEVELS = sorted(LEVELS + [500])


def test_smoothing_cells():
    # 0.1 deg IFS: 11.1 km along latitude, 3 cells for 25 km; along
    # longitude the cells narrow towards the pole: 3 cells at 40N, 5 at 56N
    # and 7 at 72N
    cell_km = ri.KM_PER_DEGREE * 0.1
    assert ri.smoothing_cells(cell_km, 25.0) == 3
    cell_km = cell_km * np.cos(np.deg2rad([40.0, 56.0, 72.0]))
    np.testing.assert_array_equal(ri.smoothing_cells(cell_km, 25.0), [3, 5, 7])
    np.testing.assert_array_equal(ri.smoothing_cells(cell_km, 0.0), [1, 1, 1])


def test_smooth_latlon():
    n_lon = np.array([3] * 20 + [5] * 20)
    yy, xx = np.meshgrid(np.arange(40.0), np.arange(50.0), indexing="ij")
    # constants are unchanged, linear fields are preserved away from the edges
    np.testing.assert_allclose(
        ri.smooth_latlon(np.full((40, 50), 3.0), 3, n_lon), 3.0
    )
    linear = 2.0 * xx - yy
    np.testing.assert_allclose(
        ri.smooth_latlon(linear, 3, n_lon)[1:-1, 2:-2], linear[1:-1, 2:-2]
    )
    # a spike is spread over 3 rows and, along each row, over that row's
    # n_lon cells
    spike = np.zeros((40, 50))
    spike[5, 10] = spike[30, 25] = 15.0
    smoothed = ri.smooth_latlon(spike, 3, n_lon)
    np.testing.assert_allclose(smoothed[4:7, 9:12], 15.0 / 9)
    np.testing.assert_allclose(smoothed[29:32, 23:28], 15.0 / 15)
    assert np.count_nonzero(smoothed) == 9 + 15
    # 1 cell is a no-op
    np.testing.assert_array_equal(
        ri.smooth_latlon(spike, 1, np.ones(40, dtype=int)), spike
    )


def test_smooth_latlon_leading_dims_and_lazy():
    data = np.random.default_rng(2).normal(size=(3, 2, 20, 30))
    da = xr.DataArray(data, dims=("time", "level", "y", "x"))
    da = da.chunk(time=1, level=1)
    n_lon = np.array([3] * 10 + [5] * 10)
    # each 2D field is smoothed on its own, not across time/levels
    expected = np.array(
        [[ri.smooth_latlon(f2d, 3, n_lon) for f2d in f3d] for f3d in data]
    )
    np.testing.assert_array_equal(ri.smooth_latlon(data, 3, n_lon), expected)
    lazy = ri.smooth_latlon_lazy(da, 3, n_lon)
    assert lazy.chunks is not None and lazy.dims == da.dims
    np.testing.assert_array_equal(lazy.values, expected)


def _forecast_coords(
    cycles=("2026-10-01T00",), levels=IFS_LEVELS, lead_hours=range(0, 49, 3)
):
    """The coordinates of an IFS forecast, all `select_forecast` reads."""
    return xr.Dataset(
        coords=dict(
            time=np.array(cycles, dtype="datetime64[ns]"),
            prediction_timedelta=np.array(
                lead_hours, dtype="timedelta64[h]"
            ).astype("timedelta64[ns]"),
            level=levels,
        )
    )


def _hours(ds):
    return (ds.prediction_timedelta.values // np.timedelta64(1, "h")).tolist()


def test_select_forecast():
    ds = ri.select_forecast(_forecast_coords(), ANALYSIS_TIME, DURATION)
    np.testing.assert_array_equal(ds.time.values, [CYCLE])
    assert ds.level.values.tolist() == LEVELS
    # the boundary times 06, 12, 18, 00 and 06 UTC of the 18h forecast from
    # 06 UTC, from the 00 UTC cycle
    assert _hours(ds) == [6, 12, 18, 24, 30]


def test_select_forecast_latest_cycle():
    cycles = (
        "2026-09-30T18",
        "2026-10-01T00",
        "2026-10-01T06",
        "2026-10-01T12",
    )
    ds = ri.select_forecast(
        _forecast_coords(cycles=cycles), ANALYSIS_TIME, DURATION
    )
    np.testing.assert_array_equal(
        ds.time.values, [np.datetime64("2026-10-01T06", "ns")]
    )
    assert _hours(ds) == [0, 6, 12, 18, 24]


def test_select_forecast_errors():
    no_600 = [level for level in IFS_LEVELS if level != 600]
    with pytest.raises(ValueError, match=r"no \[600\] hPa"):
        ri.select_forecast(
            _forecast_coords(levels=no_600), ANALYSIS_TIME, DURATION
        )
    with pytest.raises(ValueError, match="no IFS cycle"):
        ri.select_forecast(
            _forecast_coords(cycles=("2026-10-01T12",)),
            ANALYSIS_TIME,
            DURATION,
        )


def _truth(lat, lon, level=0.0, lead_hours=0.0):
    """
    A field linear in all coordinates, which the smoothing (away from the
    edges) and the bilinear interpolation keep exactly.
    """
    return 1000.0 + 2.0 * lat - 0.5 * lon + 0.1 * level + 0.01 * lead_hours


def _write_ifs(path, lon_min, lon_max, fields=True):
    """
    A synthetic IFS forecast from 00 UTC as nwp-forecast-zarr-creator writes
    it, clipped to lat 40..72 and lon `lon_min`..`lon_max`: 0.1 deg, north to
    south, one grid point beyond each bound, ERA5 names, one field per chunk.
    The fields (a single level, a pressure level and a static one) are
    `_truth`; `fields=False` writes only the coordinates.
    """
    lat = np.round(np.arange(72.1, 40.0 - 0.15, -0.1), 6)
    lon = np.round(np.arange(lon_min - 0.1, lon_max + 0.15, 0.1), 6)
    lead_hours = np.arange(0, 37, 6)
    ds = xr.Dataset(
        coords=dict(
            time=[CYCLE],
            prediction_timedelta=lead_hours.astype("timedelta64[h]").astype(
                "timedelta64[ns]"
            ),
            level=("level", IFS_LEVELS, {"units": "hPa"}),
            latitude=lat,
            longitude=lon,
        )
    )
    if fields:
        la, lo = np.meshgrid(lat, lon, indexing="ij")
        hours = lead_hours[:, None, None]
        levels = np.array(IFS_LEVELS)[:, None, None]
        ds["2m_temperature"] = (
            ("time", "prediction_timedelta", "latitude", "longitude"),
            _truth(la, lo, lead_hours=hours)[None].astype("f4"),
            {"units": "K"},
        )
        ds["temperature"] = (
            ("time", "prediction_timedelta", "level", "latitude", "longitude"),
            np.array(
                [_truth(la, lo, levels, h).astype("f4") for h in lead_hours]
            )[None],
            {"units": "K"},
        )
        ds["land_sea_mask"] = (
            ("latitude", "longitude"),
            _truth(la, lo).astype("f4"),
            {"units": "1"},
        )
    ds.chunk(time=1, prediction_timedelta=1, level=1).to_zarr(
        path, consolidated=True
    )
    return path


def test_regrid(tmp_path):
    # clipped wide enough for the 25 km moving average (see
    # test_regrid_clip_too_narrow)
    ifs = _write_ifs(tmp_path / "ifs.zarr", -27.3, 39.8)
    ri.regrid(ifs, ANALYSIS_TIME, DURATION, tmp_path / "inputs")
    ds = xr.open_zarr(
        tmp_path / "inputs" / "boundary.zarr", decode_timedelta=True
    )

    np.testing.assert_array_equal(ds.latitude, BOUNDARY_LATS)
    np.testing.assert_array_equal(ds.longitude, BOUNDARY_LONS)
    np.testing.assert_array_equal(ds.time.values, [CYCLE])
    assert _hours(ds) == [6, 12, 18, 24, 30]
    assert ds.level.values.tolist() == LEVELS
    assert ds.level.attrs["units"] == "hPa"
    assert ds["temperature"].dims == (
        "time", "prediction_timedelta", "level", "latitude", "longitude"
    )  # fmt: skip
    assert ds["land_sea_mask"].dims == ("latitude", "longitude")
    assert {str(ds[v].dtype) for v in ds.data_vars} == {"float32"}
    assert {ds[v].attrs["units"] for v in ds.data_vars} == {"K", "1"}
    assert ds.attrs["smoothing_km"] == 25.0

    lon, lat = np.meshgrid(BOUNDARY_LONS, BOUNDARY_LATS)
    hours = np.array(_hours(ds))[:, None, None]
    levels = np.array(LEVELS)[:, None, None]
    np.testing.assert_allclose(
        ds["2m_temperature"].values[0],
        _truth(lat, lon, lead_hours=hours),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        ds["temperature"].values[0],
        [_truth(lat, lon, levels, h) for h in hours[:, 0, 0]],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        ds["land_sea_mask"].values, _truth(lat, lon), rtol=1e-6
    )


def test_regrid_clip_too_narrow(tmp_path):
    # clipped to the bounding box (with one grid point beyond it): enough
    # for the bilinear interpolation, not for the 25 km moving average, which
    # needs 1 grid point beyond it along latitude and up to 3 (7 cells at
    # 72N) along longitude
    ifs = _write_ifs(tmp_path / "ifs.zarr", -27.0, 39.5, fields=False)
    with pytest.raises(ValueError, match="needs 1 .* up to 3 along longitude"):
        ri.regrid(ifs, ANALYSIS_TIME, DURATION, tmp_path / "inputs")
    # without smoothing the same clip is enough
    ri.regrid(ifs, ANALYSIS_TIME, DURATION, tmp_path / "inputs", 0.0)


def test_regrid_bounding_box_not_covered(tmp_path):
    ifs = _write_ifs(tmp_path / "ifs.zarr", -27.3, 30.0, fields=False)
    with pytest.raises(ValueError, match="outside the source grid"):
        ri.regrid(ifs, ANALYSIS_TIME, DURATION, tmp_path / "inputs", 0.0)
