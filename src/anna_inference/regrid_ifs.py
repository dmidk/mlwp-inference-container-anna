"""
Regrid an IFS forecast to the boundary input ANNA expects: a regular 0.25 deg
lat/lon bounding box with ERA5 (WeatherBench2) variable names and conventions,
following the contract in `anna_inference/boundary/ifs_7deg_model1_config.yaml`,
written as `{output}/boundary.zarr` (read through the `ifs_7deg_model1_config`
datastore config). This is the IFS counterpart of the DINI boundary in
`regrid_dini`.

The input is one IFS cycle as written by nwp-forecast-zarr-creator
(`ifs.zarr`): ERA5 variable names, dims `time` (cycle), `prediction_timedelta`,
`level` (hPa), `latitude` and `longitude`, on a regular lat/lon grid (0.1 deg),
clipped to around the bounding box. Winds on a lat/lon grid are
earth-relative, as in ERA5, so they are not rotated.

Only what the forecast uses is regridded: the latest cycle at or before the
analysis time, ANNA's pressure levels, and the lead times whose valid times are
among the forecast's boundary times (`create_inference_dataset.boundary_times`).

As for DINI, each field is smoothed with a moving average over
`--boundary-smoothing-km` before it is sampled bilinearly on the 0.25 deg
bounding box, to bring it to the scale of the ERA5 boundary ANNA was trained
with. The moving-average window is set in km, so on the lat/lon grid it spans
more grid cells along longitude towards the pole. The IFS data must reach far
enough beyond the bounding box for the moving average around every boundary
point, otherwise the regridding fails (see `check_smoothing_margin`).

Usage:
    python -m anna_inference.regrid_ifs \\
        --ifs-path s3://harmonie-zarr/ifs-dini/2026-10-01T000000Z/ifs.zarr \\
        --analysis-time 2026-10-01T06:00 --forecast-duration PT18H \\
        --output inference_workdir/inputs
"""
import argparse
import time
from pathlib import Path

import isodate
import numpy as np
import scipy.ndimage
import xarray as xr
from loguru import logger

from anna_inference.create_inference_dataset import (
    _parse_analysis_time,
    boundary_times,
)
from anna_inference.regrid_common import (
    BOUNDARY_LATS,
    BOUNDARY_LONS,
    LEVELS,
    BilinearInterpolator,
)

# km per degree of latitude (along a meridian), on a sphere with the Earth's
# mean radius
KM_PER_DEGREE = 6371.0 * np.pi / 180


def smoothing_cells(cell_km, smoothing_km):
    """
    The odd number of grid cells `cell_km` wide (a scalar or an array) whose
    width is closest to `smoothing_km`, so that the moving average is centred;
    1 (no smoothing) if `smoothing_km` is 0.
    """
    if smoothing_km <= 0:
        return np.ones(np.shape(cell_km), dtype=int)
    n = 2 * np.round((smoothing_km / np.asarray(cell_km) - 1) / 2) + 1
    return np.maximum(n, 1).astype(int)


def smooth_latlon(field, n_lat, n_lon):
    """
    Moving average over `n_lat` grid cells along latitude and, around a point
    in row j, `n_lon[j]` cells along longitude (edges: nearest value), to
    bring IFS fields to the scale of the 0.25 deg ERA5 grid before sampling
    them there. As grid cells get narrower towards the pole, `n_lon` grows
    with latitude to keep the window about the same width in km.

    Smooths over the last two (latitude, longitude) axes only.
    """
    field = np.asarray(field, dtype="f8")
    field = scipy.ndimage.uniform_filter1d(
        field, size=n_lat, axis=-2, mode="nearest"
    )
    out = np.empty_like(field)
    for n in np.unique(n_lon):
        rows = n_lon == n
        out[..., rows, :] = scipy.ndimage.uniform_filter1d(
            field[..., rows, :], size=n, axis=-1, mode="nearest"
        )
    return out


def smooth_latlon_lazy(da, n_lat, n_lon):
    """`smooth_latlon` of a (lazy) DataArray with (y, x) dims, block by block."""
    return xr.apply_ufunc(
        smooth_latlon,
        da,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        kwargs=dict(n_lat=n_lat, n_lon=n_lon),
        dask="parallelized",
        output_dtypes=[np.float64],
    )


def select_forecast(ds, analysis_time, forecast_duration):
    """
    The part of an IFS forecast an ANNA forecast from `analysis_time` uses,
    so that only that is read and regridded: ANNA's pressure levels, the
    latest cycle at or before `analysis_time` (as `boundary_to_valid_time`
    picks it) and the lead times whose valid times are among the boundary
    times the forecast uses (`boundary_times`). A missing boundary time is
    left to `boundary_to_valid_time` to report.
    """
    missing = sorted(set(LEVELS) - set(ds.level.values.tolist()))
    if missing:
        raise ValueError(
            f"the IFS forecast has no {missing} hPa level(s) (has "
            f"{ds.level.values.tolist()}); nwp-forecast-zarr-creator "
            "v0.10.0 and earlier don't write 600 hPa"
        )
    cycles = ds.time.values[ds.time.values <= np.datetime64(analysis_time)]
    if cycles.size == 0:
        raise ValueError(
            f"no IFS cycle at or before the analysis time {analysis_time} "
            f"(have {ds.time.values})"
        )
    cycle = cycles.max()
    used, _ = boundary_times(analysis_time, forecast_duration)
    valid_times = cycle + ds.prediction_timedelta.values
    lead_times = ds.prediction_timedelta.values[
        np.isin(valid_times, np.array(used, dtype="datetime64[ns]"))
    ]
    return ds.sel(time=[cycle], level=LEVELS, prediction_timedelta=lead_times)


def check_smoothing_margin(interp, n_lat, n_lon, n_x):
    """
    Fail if the moving average for any boundary point would reach beyond the
    IFS data (where `smooth_latlon` repeats the edge values instead). `interp`
    samples the boundary points from the IFS grid of `n_lon.size` rows
    (latitudes) and `n_x` columns (longitudes).
    """
    h_lat, h_lon = n_lat // 2, (n_lon // 2)[:, None]
    j, i = np.arange(n_lon.size)[:, None], np.arange(n_x)[None, :]
    complete = (
        (j >= h_lat)
        & (j < n_lon.size - h_lat)
        & (i >= h_lon)
        & (i < n_x - h_lon)
    )
    # the points whose bilinear weights fall (partly) on an incomplete window
    n_short = int((interp(complete.astype("f8")) < 1 - 1e-6).sum())
    if n_short > 0:
        raise ValueError(
            f"for {n_short} of {interp.inside.size} boundary points the "
            "moving-average window reaches beyond the IFS data: it needs "
            f"{h_lat} grid cell(s) of IFS data beyond the bounding box along "
            f"latitude and up to {h_lon.max()} along longitude. Clip the IFS "
            "data wider, or smooth less (--boundary-smoothing-km)"
        )


def regrid(
    ifs_path,
    analysis_time,
    forecast_duration,
    output,
    boundary_smoothing_km=25.0,
):
    ds = select_forecast(
        xr.open_zarr(ifs_path, decode_timedelta=True),
        analysis_time,
        forecast_duration,
    )
    lead_hours = ds.prediction_timedelta.values // np.timedelta64(1, "h")
    logger.info(
        f"IFS cycle {str(ds.time.values[0])[:16]} from {ifs_path}, lead "
        f"times {lead_hours.tolist()} h, output to {output}"
    )
    # (y, x) as BilinearInterpolator and smooth_latlon_lazy expect them
    ds = ds.rename(latitude="y", longitude="x")
    lat, lon = ds.y.values, ds.x.values
    lon2d, lat2d = np.meshgrid(BOUNDARY_LONS, BOUNDARY_LATS)
    interp = BilinearInterpolator(lon, lat, lon2d, lat2d, outside="error")
    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    n_lat = int(smoothing_cells(KM_PER_DEGREE * dlat, boundary_smoothing_km))
    n_lon = smoothing_cells(
        KM_PER_DEGREE * dlon * np.cos(np.deg2rad(lat)), boundary_smoothing_km
    )
    check_smoothing_margin(interp, n_lat, n_lon, lon.size)
    logger.info(
        f"boundary: smoothing over {n_lat} x {n_lon.min()}..{n_lon.max()} "
        "IFS grid cells (latitude x longitude) before sampling"
    )

    out_dims = ("latitude", "longitude")
    ds_b = xr.Dataset(
        {
            v: interp.apply(smooth_latlon_lazy(ds[v], n_lat, n_lon), out_dims)
            .astype("f4")
            .assign_attrs(ds[v].attrs)
            for v in ds.data_vars
            if {"y", "x"} <= set(ds[v].dims)
        },
        coords=dict(
            latitude=("latitude", BOUNDARY_LATS),
            longitude=("longitude", BOUNDARY_LONS),
        ),
    )
    ds_b.attrs = dict(
        source=str(ifs_path),
        description=(
            "IFS regridded to the ERA5 0.25 deg bounding box in the IFS "
            "boundary contract layout (regrid_ifs.py)"
        ),
        smoothing_km=boundary_smoothing_km,
    )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    logger.info("reading and regridding IFS")
    start = time.perf_counter()
    ds_b = ds_b.compute()
    logger.info(f"regridded in {time.perf_counter() - start:.0f} s")
    ds_b.to_zarr(output / "boundary.zarr", mode="w", consolidated=True)
    logger.info(f"wrote {output / 'boundary.zarr'}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ifs-path", required=True, help=".../ifs-dini/{analysis}/ifs.zarr"
    )
    parser.add_argument(
        "--analysis-time",
        required=True,
        type=_parse_analysis_time,
        help="ISO8601, UTC if no timezone",
    )
    parser.add_argument(
        "--forecast-duration",
        required=True,
        type=isodate.parse_duration,
        help="ISO8601 duration, e.g. PT18H",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--boundary-smoothing-km",
        type=float,
        default=25.0,
        help=(
            "smooth IFS with a moving average over this scale before "
            "sampling the 0.25 deg boundary (0: off)"
        ),
    )
    args = parser.parse_args()

    regrid(
        ifs_path=args.ifs_path,
        analysis_time=args.analysis_time,
        forecast_duration=args.forecast_duration,
        output=args.output,
        boundary_smoothing_km=args.boundary_smoothing_km,
    )


if __name__ == "__main__":
    main()
