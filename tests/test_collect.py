"""
Tests for `anna_inference.artifact.collect` (collecting the inference
artifact on the training machine) and the artifact layout it shares with
`anna_inference.artifact.assemble`.

The training setup is synthetic: the repository's training datastore
configs, with small datastore zarrs laid out as mllam-data-prep writes them
(`grid_index` stacked from x/y for the interior, lat/lon per point for the
boundary, statistics as `{category}__train__{statistic}`).

    uv run pytest tests
"""
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from anna_inference import model_configs as mc
from anna_inference.artifact import collect, layout

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
SPLITS = xr.DataArray(
    np.array([["2000-01-01", "2000-01-02"]] * 3),
    dims=["split_name", "split_part"],
    coords=dict(
        split_name=["train", "val", "test"], split_part=["start", "end"]
    ),
)


def _stats(category, features):
    n = len(features)
    return {
        f"{category}__train__mean": (
            f"{category}_feature",
            np.arange(n) * 1.0,
        ),
        f"{category}__train__std": (f"{category}_feature", np.arange(n) + 1.0),
    }


def _interior_datastore(nx=4, ny=3):
    """Interior datastore, grid_index stacked from [x, y] as in training."""
    xs, ys = 2500.0 * np.arange(nx), 2500.0 * np.arange(ny) - 1000.0
    x, y = np.meshgrid(xs, ys, indexing="ij")  # [x, y] stacking order
    x, y = x.ravel(), y.ravel()
    statics = ["lsm", "orography"]
    return xr.Dataset(
        {
            "static": (
                ("static_feature", "grid_index"),
                np.stack([x / 1e4, y + 2.0 * x]),
            ),
            "splits": SPLITS,
            **_stats("state", ["t2m", "u10m"]),
            **_stats("static", statics),
        },
        coords=dict(
            grid_index=np.arange(x.size),
            x=("grid_index", x, dict(standard_name="projection_x_coordinate")),
            y=("grid_index", y, dict(standard_name="projection_y_coordinate")),
            lat=("grid_index", 50.0 + y / 1e5),
            lon=("grid_index", 10.0 + x / 1e5),
            static_feature=statics,
            static_feature_units=("static_feature", ["1", "m**2 s**-2"]),
            state_feature=["t2m", "u10m"],
        ),
    )


def _boundary_datastore():
    forcings = ["mean_sea_level_pressure", "2m_temperature", "toa_radiation"]
    statics = ["land_sea_mask", "geopotential_at_surface"]
    grid_index = np.array([431, 432, 700, 1001, 34357])
    return xr.Dataset(
        {
            "static": (
                ("static_feature", "grid_index"),
                np.ones((2, grid_index.size)),
            ),
            "splits": SPLITS,
            **_stats("forcing", forcings),
            **_stats("static", statics),
        },
        coords=dict(
            grid_index=grid_index,
            latitude=("grid_index", np.linspace(40.5, 71.5, grid_index.size)),
            longitude=("grid_index", np.linspace(-26, 39.5, grid_index.size)),
            forcing_feature=forcings,
            static_feature=statics,
        ),
    )


@pytest.fixture
def training_setup(tmp_path):
    """
    A training directory as on the training machine: neural-lam config,
    datastore configs and their zarrs, and a checkpoint.
    """
    training = mc.model_configs(CONFIGS_DIR, mc.TRAINING_BOUNDARY_SOURCE)
    train_dir = tmp_path / "training"
    (train_dir / "datastores").mkdir(parents=True)
    for fn, ds in [
        (training.interior_datastore, _interior_datastore()),
        (training.boundary_datastore, _boundary_datastore()),
    ]:
        shutil.copy(CONFIGS_DIR / fn, train_dir / "datastores" / fn)
        ds.to_zarr(train_dir / "datastores" / Path(fn).with_suffix(".zarr"))
    nl_config = yaml.safe_load(
        (CONFIGS_DIR / training.neural_lam_config).read_text()
    )
    # as in training: one relative and one absolute datastore path
    nl_config["datastore"][
        "config_path"
    ] = f"datastores/{training.interior_datastore}"
    nl_config["datastore_boundary"]["config_path"] = str(
        train_dir / "datastores" / training.boundary_datastore
    )
    fp_nl_config = train_dir / "7deg_config.yaml"
    fp_nl_config.write_text(yaml.safe_dump(nl_config))
    fp_ckpt = train_dir / "model.ckpt"
    fp_ckpt.write_bytes(b"not really a checkpoint")
    return fp_nl_config, fp_ckpt


def test_find_datastore_paths_with_boundary(training_setup):
    fp_nl_config, _ = training_setup
    paths = collect.find_datastore_paths(fp_nl_config)
    datastores = fp_nl_config.parent / "datastores"
    assert paths == {
        ("datastore",): datastores / "danra_model1_config.yaml",
        ("datastore_boundary",): datastores / "era_7deg_model1_config.yaml",
    }


def test_find_datastore_paths_multiple_datastores(tmp_path):
    nl_config = dict(
        datastores=dict(
            danra=dict(kind="mdp", config_path="danra.yaml"),
            era5=dict(kind="mdp", config_path="/data/era5.yaml"),
        )
    )
    fp_nl_config = tmp_path / "config.yaml"
    fp_nl_config.write_text(yaml.dump(nl_config))
    assert collect.find_datastore_paths(fp_nl_config) == {
        ("datastores", "danra"): tmp_path / "danra.yaml",
        ("datastores", "era5"): Path("/data/era5.yaml"),
    }


def test_find_datastore_paths_no_datastore(tmp_path):
    fp_nl_config = tmp_path / "config.yaml"
    fp_nl_config.write_text(yaml.dump(dict(training=dict())))
    with pytest.raises(ValueError):
        collect.find_datastore_paths(fp_nl_config)


def test_statistics_only():
    ds_stats = collect.statistics(_boundary_datastore())
    assert set(ds_stats.data_vars) == {
        "forcing__train__mean",
        "forcing__train__std",
        "static__train__mean",
        "static__train__std",
    }


def test_interior_grid_unstacks():
    ds = _interior_datastore(nx=4, ny=3)
    grid = collect.interior_grid(ds)
    assert dict(grid.sizes) == dict(y=3, x=4)
    assert np.all(np.diff(grid.x) > 0) and np.all(np.diff(grid.y) > 0)
    # every point back where its x/y say
    X, Y = np.meshgrid(grid.x, grid.y)
    np.testing.assert_array_equal(grid.lat, 50.0 + Y / 1e5)
    np.testing.assert_array_equal(grid.lon, 10.0 + X / 1e5)
    np.testing.assert_array_equal(grid.lsm, X / 1e4)
    np.testing.assert_array_equal(grid.orography, Y + 2.0 * X)
    assert grid.orography.attrs["units"] == "m**2 s**-2"
    assert grid.x.attrs == ds.x.attrs


def test_collect(training_setup, tmp_path):
    fp_nl_config, fp_ckpt = training_setup
    out = tmp_path / "artifact"
    collect.collect(fp_nl_config, fp_ckpt, out, CONFIGS_DIR, "test")

    # the layout (checked by collect itself too), with the checkpoint
    for fp in layout.required_paths(out, checkpoint=True):
        assert fp.exists(), fp
    assert layout.checkpoint_path(out).read_bytes() == fp_ckpt.read_bytes()

    # the training neural-lam config, with sibling datastore paths
    training = layout.training_configs(out)
    nl_config = yaml.safe_load(
        (out / "configs" / training.neural_lam_config).read_text()
    )
    assert nl_config["datastore"]["config_path"] == "danra_model1_config.yaml"
    assert (
        nl_config["datastore_boundary"]["config_path"]
        == "era_7deg_model1_config.yaml"
    )

    # statistics and grids from the training datastores
    ds_stats = xr.open_zarr(
        layout.stats_path(out, training.boundary_datastore)
    )
    xr.testing.assert_identical(
        ds_stats.forcing__train__std.load(),
        _boundary_datastore().forcing__train__std,
    )
    ds_grid = xr.open_zarr(layout.grid_path(out, training.boundary_datastore))
    np.testing.assert_array_equal(
        ds_grid.grid_index, _boundary_datastore().grid_index
    )
    assert "splits" in xr.open_zarr(
        layout.stats_datastore_path(out, training.boundary_datastore)
    )

    # a new checkpoint: its md5 is recorded, without a download URL
    model = yaml.safe_load((out / "configs" / "model.yaml").read_text())
    assert model["checkpoint"]["url"] is None
    assert yaml.safe_load((out / "artifact.yaml").read_text())[
        "checkpoint_included"
    ]


def test_collect_rejects_other_datastores(training_setup, tmp_path):
    fp_nl_config, fp_ckpt = training_setup
    nl_config = yaml.safe_load(fp_nl_config.read_text())
    nl_config["datastore"]["config_path"] = "datastores/other.yaml"
    fp_nl_config.write_text(yaml.safe_dump(nl_config))
    with pytest.raises(ValueError, match="training datastore"):
        collect.collect(
            fp_nl_config, fp_ckpt, tmp_path / "artifact", CONFIGS_DIR, "test"
        )
