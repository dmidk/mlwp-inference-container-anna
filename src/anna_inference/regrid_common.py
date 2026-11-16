"""
The boundary grid, the pressure levels and the bilinear interpolation shared
by the regridding modules.
"""
import numpy as np
import xarray as xr
from loguru import logger

LEVELS = [100, 200, 400, 600, 700, 850, 925, 1000]  # hPa, as used by ANNA

# ERA5 (WeatherBench2) 0.25 deg box covering the boundary points ANNA was
# trained with (lat 40.50..71.50, lon -26.25..39.50), as in the contract in
# anna_inference/boundary/ifs_7deg_model1_config.yaml. The east edge must be
# exactly 39.5E, the edge of the training ERA5 subset: the cropping margin
# reaches further east, so a wider box would add boundary points not used in
# training
BOUNDARY_LATS = np.arange(72.0, 40.0 - 0.125, -0.25)
BOUNDARY_LONS = np.arange(-27.0, 39.5 + 0.125, 0.25)


class BilinearInterpolator:
    """
    Bilinear interpolation from a regular (y, x) grid to arbitrary target
    points given in the same projected coordinates. Weights are computed once
    and applied to any number of fields.
    """

    def __init__(self, x_src, y_src, x_tgt, y_tgt, outside="error"):
        dx, dy = x_src[1] - x_src[0], y_src[1] - y_src[0]
        fi = (np.asarray(x_tgt) - x_src[0]) / dx
        fj = (np.asarray(y_tgt) - y_src[0]) / dy
        self.shape = np.shape(x_tgt)
        self.inside = (
            (fi >= 0)
            & (fi <= x_src.size - 1)
            & (fj >= 0)
            & (fj <= y_src.size - 1)
        )
        n_outside = int((~self.inside).sum())
        if n_outside > 0:
            msg = (
                f"{n_outside} of {self.inside.size} target points are outside "
                "the source grid"
            )
            if outside == "error":
                raise ValueError(msg)
            logger.warning(
                f"{msg}, filling with the nearest source grid point"
            )
        fi = np.clip(fi, 0, x_src.size - 1)
        fj = np.clip(fj, 0, y_src.size - 1)
        self.i0 = np.minimum(np.floor(fi).astype(int), x_src.size - 2)
        self.j0 = np.minimum(np.floor(fj).astype(int), y_src.size - 2)
        self.wx = fi - self.i0
        self.wy = fj - self.j0
        self.n_outside = n_outside

    def __call__(self, field):
        """Interpolate a (..., y, x) field (the last two axes)."""
        f = np.asarray(field)
        i0, j0, wx, wy = self.i0, self.j0, self.wx, self.wy
        return (
            f[..., j0, i0] * (1 - wx) * (1 - wy)
            + f[..., j0, i0 + 1] * wx * (1 - wy)
            + f[..., j0 + 1, i0] * (1 - wx) * wy
            + f[..., j0 + 1, i0 + 1] * wx * wy
        )

    def apply(self, da, out_dims):
        """
        Interpolate a (lazy) DataArray with (y, x) dims, block by block with
        dask; the target points get dims `out_dims`.
        """
        return xr.apply_ufunc(
            self,
            da,
            input_core_dims=[["y", "x"]],
            output_core_dims=[list(out_dims)],
            dask="parallelized",
            output_dtypes=[np.float64],
            dask_gufunc_kwargs=dict(
                output_sizes=dict(zip(out_dims, self.shape))
            ),
        )
