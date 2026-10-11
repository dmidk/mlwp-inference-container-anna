"""
Collect the ANNA inference artifact on the machine the model was trained on.

This is how the artifact should be made for a newly trained model: on the
training machine, where the training datastores (the mllam-data-prep zarr
next to each datastore config) still exist. Everything that only exists
there comes from the training setup itself:

- the training neural-lam config and its datastore configs (written as the
  neural-lam config `model.yaml` names, with sibling datastore paths);
- the training statistics of the interior and boundary datastores
  (`stats/`), and the boundary "stats datastore" neural-lam opens for
  `overload_stats_path` (`configs/<boundary>.zarr`);
- the grids: the interior grid (x, y, lat, lon) and statics, unstacked from
  the interior datastore, and the lat/lon of the boundary points
  (`grids/`);
- the checkpoint.

The model description (`model.yaml`) comes from `artifact/configs/` of this
repository; its training configs there are the reference for the datastore
names. The artifact describes the model as trained: the operational (DINI,
IFS) boundary configs are part of this package, not of the artifact. The
result has the layout of
`anna_inference.artifact.layout` (checked at the end), the same as
`anna_inference.artifact.assemble` reconstructs for the paper's model, whose
training datastores were no longer available.

Ported from `mlwm.build_inference_artifact` (mlwm-deployment), which
collected checkpoint, configs and statistics but no grids, stats datastore or
model description, and uploaded to S3.

Usage, on the training machine:

    uv run python -m anna_inference.artifact.collect \\
        --nl-config <training neural-lam config, e.g. 7deg_config.yaml> \\
        --checkpoint <checkpoint.ckpt> \\
        --artifact-name <name> [--zip]

writes `artifact/build/<name>/` (with the checkpoint) and, with `--zip`,
`artifact/build/<name>.zip` (without it).
"""
import argparse
import datetime
import hashlib
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from loguru import logger

from anna_inference import model_configs as mc
from anna_inference.artifact import layout

# the repository (the package is installed in editable mode for development)
REPO_DIR = Path(__file__).resolve().parents[3]
BUILD_DIR = REPO_DIR / "artifact" / "build"
REPO_URL = "https://github.com/leifdenby/mlwp-inference-container-anna"


def find_datastore_paths(nl_config_path):
    """
    Paths of the datastore configs a neural-lam config names, keyed by
    their entry in the config: `("datastore",)` (interior),
    `("datastore_boundary",)` and `("datastores", name)`. Relative paths are
    resolved against the neural-lam config's directory, as neural-lam does.
    """
    nl_config_path = Path(nl_config_path)
    nl_config = yaml.safe_load(nl_config_path.read_text())

    def _resolve(path):
        path = Path(path)
        return path if path.is_absolute() else nl_config_path.parent / path

    paths = {}
    for key in ["datastore", "datastore_boundary"]:
        if nl_config.get(key) is not None:
            paths[(key,)] = _resolve(nl_config[key]["config_path"])
    for name, entry in (nl_config.get("datastores") or {}).items():
        paths[("datastores", name)] = _resolve(entry["config_path"])
    if not paths:
        raise ValueError(
            f"no datastore in {nl_config_path}, is it a neural-lam config?"
        )
    return paths


def open_datastore(datastore_config_path):
    """The datastore (zarr) mllam-data-prep created next to its config."""
    fp = Path(datastore_config_path).with_suffix(".zarr")
    if not fp.exists():
        raise FileNotFoundError(
            f"{fp}: the training datastore isn't there (collect on the "
            "machine the model was trained on)"
        )
    return xr.open_zarr(fp)


def statistics(ds):
    """
    The statistics in a datastore: mllam-data-prep names them
    `{category}__{split}__{statistic}`, e.g. `state__train__mean`.
    """
    names = [v for v in ds.data_vars if len(str(v).split("__")) == 3]
    ds_stats = ds[names].compute()
    ds_stats.attrs.update(ds.attrs)
    return ds_stats


def interior_grid(ds):
    """
    The interior grid as (y, x) fields: x, y, lat, lon and the statics,
    unstacked from the datastore's `grid_index` (whatever order mdp stacked
    x and y in).
    """
    xs, ix = np.unique(ds.x.values, return_inverse=True)
    ys, iy = np.unique(ds.y.values, return_inverse=True)
    if xs.size * ys.size != ds.grid_index.size:
        raise ValueError("the interior grid_index isn't a full x/y grid")

    def _unstack(values):
        field = np.full((ys.size, xs.size), np.nan)
        field[iy, ix] = values
        return field

    ds_grid = xr.Dataset(
        coords=dict(
            x=("x", xs, dict(ds.x.attrs)),
            y=("y", ys, dict(ds.y.attrs)),
            lat=(
                ("y", "x"),
                _unstack(ds.lat.values),
                dict(standard_name="latitude", units="degrees_north"),
            ),
            lon=(
                ("y", "x"),
                _unstack(ds.lon.values),
                dict(standard_name="longitude", units="degrees_east"),
            ),
        )
    )
    if "static" in ds:
        static = ds.static.load()
        for feature in static.static_feature.values:
            da = static.sel(static_feature=feature)
            attrs = {}
            if "static_feature_units" in static.coords:
                attrs["units"] = str(da.static_feature_units.values)
            ds_grid[str(feature)] = (("y", "x"), _unstack(da.values), attrs)
    return ds_grid


def boundary_grid(ds):
    """lat/lon of the boundary points, by `grid_index`."""
    return xr.Dataset(
        coords=dict(
            grid_index=("grid_index", ds.grid_index.values),
            latitude=("grid_index", ds.latitude.values),
            longitude=("grid_index", ds.longitude.values),
        )
    )


def _md5(fp):
    md5 = hashlib.md5()
    with open(fp, "rb") as fh:
        for chunk in iter(lambda: fh.read(2**20), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _git_describe():
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _versions():
    import mllam_data_prep
    import neural_lam

    return dict(
        python=sys.version.split()[0],
        mllam_data_prep=mllam_data_prep.__version__,
        neural_lam=getattr(neural_lam, "__version__", "unknown"),
    )


def collect(nl_config, checkpoint, out, configs_dir, artifact_name):
    """Collect the artifact into the (new) directory `out`."""
    out = Path(out)
    configs_out = out / layout.CONFIGS
    configs_out.mkdir(parents=True)

    # the model description from this repository; the training configs
    # there are the reference for the datastore names the deployment expects
    shutil.copy(Path(configs_dir) / "model.yaml", configs_out / "model.yaml")
    training = mc.training_configs(configs_dir)

    # the training configs from the training machine, named as in model.yaml
    # and the reference training config
    paths = find_datastore_paths(nl_config)
    for key in [("datastore",), ("datastore_boundary",)]:
        if key not in paths:
            raise ValueError(f"{nl_config} has no {key[0]}")
    expected = {
        ("datastore",): training.interior_datastore,
        ("datastore_boundary",): training.boundary_datastore,
    }
    nl = yaml.safe_load(Path(nl_config).read_text())
    for key, fn in expected.items():
        if mc.TrainingConfigs.name(paths[key]) != mc.TrainingConfigs.name(fn):
            raise ValueError(
                f"{key[0]} of {nl_config} is {paths[key].name}, but the "
                f"configs in {configs_dir} use {fn} for the training "
                "datastore"
            )
        logger.info(f"training datastore config {paths[key]}")
        shutil.copy(paths[key], configs_out / fn)
        nl[key[0]]["config_path"] = fn
    (configs_out / training.neural_lam_config).write_text(
        yaml.safe_dump(nl, sort_keys=False)
    )

    # statistics, the boundary stats datastore and the grids, from the
    # training datastores
    ds_interior = open_datastore(paths[("datastore",)])
    ds_boundary = open_datastore(paths[("datastore_boundary",)])
    for fn, ds in [
        (training.interior_datastore, ds_interior),
        (training.boundary_datastore, ds_boundary),
    ]:
        fp = layout.stats_path(out, fn)
        fp.parent.mkdir(exist_ok=True)
        logger.info(f"statistics -> {fp}")
        statistics(ds).to_zarr(fp, consolidated=True)
    # (written after the configs: neural-lam warns if it's older than them)
    fp = layout.stats_datastore_path(out, training.boundary_datastore)
    logger.info(f"boundary stats datastore -> {fp}")
    ds_stats_datastore = statistics(ds_boundary)
    ds_stats_datastore["splits"] = ds_boundary["splits"].load()
    ds_stats_datastore.to_zarr(fp, consolidated=True)
    for fn, grid in [
        (training.interior_datastore, interior_grid(ds_interior)),
        (training.boundary_datastore, boundary_grid(ds_boundary)),
    ]:
        fp = layout.grid_path(out, fn)
        fp.parent.mkdir(exist_ok=True)
        logger.info(f"grid -> {fp}")
        grid.to_zarr(fp, consolidated=True)

    # the checkpoint, where model.yaml says it goes
    model = yaml.safe_load((configs_out / "model.yaml").read_text())
    fp_ckpt = layout.checkpoint_path(out)
    logger.info(f"checkpoint {checkpoint} -> {fp_ckpt}")
    shutil.copyfile(checkpoint, fp_ckpt)
    md5 = _md5(fp_ckpt)
    if md5 != model["checkpoint"].get("md5"):
        # a new model: model.yaml describes the published checkpoint, which
        # this one isn't (yet)
        logger.warning(
            f"checkpoint md5 {md5} differs from model.yaml's "
            f"{model['checkpoint'].get('md5')}: recording it in the "
            "artifact's model.yaml, without a download URL"
        )
        model["checkpoint"].update(md5=md5, url=None)
        (configs_out / "model.yaml").write_text(
            yaml.safe_dump(model, sort_keys=False)
        )

    meta = dict(
        artifact_name=artifact_name,
        description=f"ANNA inference artifact, see {REPO_URL}",
        model_checkpoint=model["checkpoint"],
        collected_from=dict(
            neural_lam_config=str(Path(nl_config).resolve()),
            datastores={
                key[0]: str(Path(p).resolve()) for key, p in paths.items()
            },
            checkpoint=str(Path(checkpoint).resolve()),
            hostname=socket.gethostname(),
            cwd=os.getcwd(),
            argv=sys.argv,
            versions=_versions(),
            repo=REPO_URL,
            repo_git=_git_describe(),
        ),
        checkpoint_included=True,
        created_on=datetime.datetime.now().replace(microsecond=0).isoformat(),
    )
    (out / layout.ARTIFACT_META).write_text(
        yaml.safe_dump(meta, sort_keys=False)
    )
    shutil.copy(REPO_DIR / "artifact" / "package-README.md", out / "README.md")
    layout.check_artifact(out, checkpoint=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--nl-config",
        required=True,
        type=Path,
        help="neural-lam config the model was trained with",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="artifact directory to create (must not exist, default: "
        "artifact/build/<artifact-name>)",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=REPO_DIR / "artifact" / "configs",
        help="model.yaml and the reference training configs",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="also write the artifact, without the checkpoint, as zip",
    )
    args = parser.parse_args()
    output = args.output or BUILD_DIR / args.artifact_name
    fp_zip = output.with_suffix(".zip") if args.zip else None
    for fp in [output, fp_zip]:
        if fp is not None and fp.exists():
            raise SystemExit(f"{fp} already exists, remove it first")

    # collect in a temporary directory next to the output and move it into
    # place at the end, so a failed run doesn't leave a half-collected
    # artifact behind
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmpdir:
        out = Path(tmpdir) / "artifact"
        collect(
            args.nl_config,
            args.checkpoint,
            out,
            args.configs_dir,
            args.artifact_name,
        )
        shutil.move(str(out), str(output))
    logger.info(f"collected artifact in {output}")
    if fp_zip is not None:
        layout.write_zip(output, fp_zip)
        logger.info(f"packaged artifact (without checkpoint) as {fp_zip}")


if __name__ == "__main__":
    main()
