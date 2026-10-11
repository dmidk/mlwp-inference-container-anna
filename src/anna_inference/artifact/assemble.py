"""
Assemble the ANNA inference package: the paper's DANRA model as trained
(Zenodo 15131838, see `artifact/configs/model.yaml`; DANRA interior, ERA5
boundary), except the model checkpoint itself, which is downloaded from
Zenodo separately (`--fetch-checkpoint` for a local directory, the
Containerfile for images). The operational boundaries (DINI, IFS) are not
part of it; they are in `anna_inference/boundary/`. See artifact/README.md.

The interior statistics and the boundary grid of the datastores the model was
trained with are taken from the (superseded) `gefion-1` inference artifact,
which was built from the same datastores; the ERA5 boundary statistics are
computed with `anna_inference.artifact.era5_boundary_stats`. The package
contains:

    configs/                the training neural-lam config, the interior
                            (DANRA) and boundary (ERA5) datastore configs and
                            model.yaml (the model), from artifact/configs/ in
                            this repository
    configs/era_7deg_model1_config.zarr
                            ERA5 "stats datastore" that neural-lam opens via
                            `overload_stats_path` (statistics and splits only)
    stats/danra_model1_config.stats.zarr
                            interior training statistics (from gefion-1)
    stats/era_7deg_model1_config.stats.zarr
                            ERA5 boundary training statistics (from
                            era5_boundary_stats, or a placeholder)
    grids/era_7deg_model1_config.grid.zarr
                            lat/lon of the 18014 boundary points the model was
                            trained with, recovered from the gefion-1
                            checkpoint (same boundary datastore)
    grids/danra_model1_config.grid.zarr
                            DANRA grid (x, y, lat, lon) and statics (lsm,
                            orography) from the public DANRA v0.5.0 store, the
                            target grid for regridding DINI (regrid_dini)
    artifact.yaml           provenance of this package
    README.md               description of the package (package-README.md)
    danra_model.ckpt        the model checkpoint, only with --fetch-checkpoint
                            (never part of the zip)

Usage (from the repository root):

    uv run python -m anna_inference.artifact.assemble \\
        --gefion-1-zip gefion-1.zip \\
        --boundary-stats era_7deg_model1_config.stats.zarr \\
        --artifact-name anna-danra-2026-10-06 --zip

writes `artifact/build/<artifact-name>/` and, with `--zip`,
`artifact/build/<artifact-name>.zip` (contents at the zip root,
without the checkpoint) for publishing. Add `--fetch-checkpoint` to also
download the checkpoint (md5-checked) for running locally.
"""
import argparse
import datetime
import hashlib
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
import zarr
from loguru import logger

from anna_inference import fetch_checkpoint
from anna_inference import model_configs as mc
from anna_inference import regrid_dini
from anna_inference.artifact import (
    era5_boundary_stats,
    layout,
    sanitize_checkpoint,
)

# the repository (the package is installed in editable mode for development)
REPO_DIR = Path(__file__).resolve().parents[3]
BUILD_DIR = REPO_DIR / "artifact" / "build"
REPO_URL = "https://github.com/leifdenby/mlwp-inference-container-anna"

BOUNDARY_DATASTORE_NAME = "era_7deg_model1_config"
INTERIOR_DATASTORE_NAME = "danra_model1_config"
N_BOUNDARY_POINTS = 18014
BOUNDARY_STATS_DESCRIPTION = (
    "Training statistics of the ERA5 boundary datastore "
    "(era_7deg_model1_config) of the DANRA ML LAM models "
    "(arXiv:2504.09340), recomputed from WeatherBench2 ERA5 with "
    f"anna_inference.artifact.era5_boundary_stats ({REPO_URL})"
)


def _check_model_configs(configs_dir):
    """
    Check that the training datastores this package provides statistics and
    grids for (from gefion-1 and the ERA5 statistics) are the ones the
    training neural-lam config named in model.yaml uses.
    """
    training = mc.training_configs(configs_dir)
    names = dict(
        interior=(training.interior_datastore, INTERIOR_DATASTORE_NAME),
        boundary=(training.boundary_datastore, BOUNDARY_DATASTORE_NAME),
    )
    for what, (fn_config, expected) in names.items():
        if mc.TrainingConfigs.name(fn_config) != expected:
            raise SystemExit(
                f"{what} datastore of the configs is {fn_config}, but this "
                f"package provides the statistics and grids of {expected}"
            )


def _sha256(fp):
    h = hashlib.sha256()
    with open(fp, "rb") as fh:
        for chunk in iter(lambda: fh.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _boundary_grid_from_checkpoint(ckpt, subset_lats, subset_lons):
    """
    Recover the lat/lon of the boundary points ANNA was trained with.

    The checkpoint's pickled training boundary datastore keeps its
    `grid_index` index in memory. mllam-data-prep stacked `[longitude,
    latitude]` of the ERA5 subset (latitude descending, longitude sorted
    ascending in 0..360) into `grid_index`, so each index is
    `lon_idx * n_lat + lat_idx`.
    """
    datastore = ckpt["hyper_parameters"]["datastore_boundary"]
    state = vars(datastore).get("_state", vars(datastore))
    grid_index = np.asarray(state["_ds"].grid_index.values)

    lats = np.asarray(subset_lats)
    lons = np.sort(np.asarray(subset_lons))  # as stored in the ERA5 subset
    lon_idx, lat_idx = np.divmod(grid_index, lats.size)
    if lon_idx.max() >= lons.size:
        raise ValueError(
            "boundary grid_index doesn't fit the ERA5 subset grid"
        )
    if grid_index.size != N_BOUNDARY_POINTS:
        raise ValueError(
            f"expected {N_BOUNDARY_POINTS} boundary points, got {grid_index.size}"
        )
    return xr.Dataset(
        coords=dict(
            grid_index=("grid_index", grid_index),
            latitude=("grid_index", lats[lat_idx]),
            longitude=("grid_index", lons[lon_idx]),
        ),
        attrs=dict(
            description=(
                "lat/lon of the ERA5 (0.25 deg) boundary points the DANRA "
                "ML LAM models (arXiv:2504.09340) were trained with, "
                "recovered from the pickled boundary datastore grid_index of "
                "the gefion-1 checkpoint (same datastore) and the "
                "era_danra_model1_subset grid (grid_index stacked "
                "[longitude, latitude])"
            )
        ),
    )


def _create_stats_datastore(ds_stats, fp_boundary_config, fp_out):
    """
    Create the zarr neural-lam opens as boundary "stats datastore" (via
    `overload_stats_path`): the statistics, their feature coordinates and the
    `splits` of the boundary datastore config. neural-lam only uses the
    `{forcing,static}__train__{mean,std}` of it.
    """
    import mllam_data_prep as mdp

    config = mdp.Config.from_yaml_file(fp_boundary_config)
    splits = config.output.splitting.splits
    ds = ds_stats.copy()
    ds["splits"] = xr.DataArray(
        np.array([[str(s.start), str(s.end)] for s in splits.values()]),
        dims=["split_name", "split_part"],
        coords=dict(split_name=list(splits), split_part=["start", "end"]),
    )
    ds.to_zarr(fp_out, mode="w", consolidated=True)


def _check_stats_datastore(fp_boundary_config):
    """Open the stats datastore as neural-lam will and check the statistics."""
    from neural_lam.datastore.mdp import MDPDatastore

    fp_zarr = Path(fp_boundary_config).with_suffix(".zarr")
    if not fp_zarr.exists():
        # MDPDatastore would otherwise try to build the full training dataset
        raise FileNotFoundError(fp_zarr)
    datastore = MDPDatastore(config_path=fp_boundary_config)
    for category, n_features in [("forcing", 58), ("static", 2)]:
        ds = datastore.get_standardization_dataarray(category=category)
        for op in ["mean", "std"]:
            values = ds[f"{category}_{op}"].values
            if values.size != n_features or not np.isfinite(values).all():
                raise ValueError(f"bad {category}_{op} in stats datastore")
        if not (ds[f"{category}_std"].values > 0).all():
            raise ValueError(f"non-positive {category}_std in stats datastore")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gefion-1-zip", required=True, type=Path)
    parser.add_argument(
        "--boundary-stats",
        required=True,
        type=Path,
        help=f"{BOUNDARY_DATASTORE_NAME}.stats.zarr with the ERA5 boundary stats",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="artifact directory to create (must not exist, default: "
        "artifact/build/<artifact-name>)",
    )
    parser.add_argument(
        "--configs-dir", type=Path, default=REPO_DIR / "artifact" / "configs"
    )
    parser.add_argument(
        "--artifact-name",
        default="anna-local",
        help="name of the artifact (written to artifact.yaml)",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="also write the artifact as zip, next to the artifact directory",
    )
    parser.add_argument(
        "--fetch-checkpoint",
        action="store_true",
        help="also download the model checkpoint into the artifact directory",
    )
    parser.add_argument(
        "--allow-placeholder-stats",
        action="store_true",
        help="allow zipping an artifact with placeholder boundary statistics",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = BUILD_DIR / args.artifact_name
    args.zip = args.output.with_suffix(".zip") if args.zip else None

    if args.output.exists():
        raise SystemExit(f"{args.output} already exists, remove it first")
    if args.zip is not None and args.zip.exists():
        raise SystemExit(f"{args.zip} already exists, remove it first")

    _check_model_configs(args.configs_dir)

    ds_boundary_stats = xr.open_zarr(args.boundary_stats).load()
    is_placeholder = ds_boundary_stats.attrs.get("PLACEHOLDER") == "true"
    if is_placeholder:
        if args.zip is not None and not args.allow_placeholder_stats:
            raise SystemExit(
                "refusing to package an artifact zip with PLACEHOLDER boundary "
                "statistics (use --allow-placeholder-stats to override)"
            )
        logger.warning(
            "the boundary statistics are a PLACEHOLDER, replace them with the "
            "output of era5_boundary_stats for real forecasts"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        logger.info(f"unpacking {args.gefion_1_zip}")
        with zipfile.ZipFile(args.gefion_1_zip) as zf:
            zf.extractall(tmpdir)

        # build in a temporary directory and move into place at the end, so
        # that a failed run doesn't leave a half-assembled artifact behind
        out = tmpdir / "artifact"
        (out / "stats").mkdir(parents=True)
        (out / "grids").mkdir()
        (out / "configs").mkdir()

        logger.info(
            "recovering the boundary grid from the gefion-1 checkpoint"
        )
        ckpt = sanitize_checkpoint.load_checkpoint(tmpdir / "checkpoint.pkl")
        ds_grid = _boundary_grid_from_checkpoint(
            ckpt,
            era5_boundary_stats.SUBSET_LATS,
            era5_boundary_stats.SUBSET_LONS,
        )
        ds_grid.to_zarr(out / "grids" / f"{BOUNDARY_DATASTORE_NAME}.grid.zarr")
        logger.info("caching DANRA grid and statics")
        regrid_dini.create_danra_grid(
            out / "grids" / f"{INTERIOR_DATASTORE_NAME}.grid.zarr"
        )

        for fp in sorted(args.configs_dir.glob("*.yaml")):
            shutil.copy(fp, out / "configs" / fp.name)
        shutil.copytree(
            tmpdir / "stats" / f"{INTERIOR_DATASTORE_NAME}.stats.zarr",
            out / "stats" / f"{INTERIOR_DATASTORE_NAME}.stats.zarr",
        )
        shutil.copytree(
            args.boundary_stats,
            out / "stats" / f"{BOUNDARY_DATASTORE_NAME}.stats.zarr",
        )
        # older runs of era5_boundary_stats described the stats as
        # being for gefion-1; they are the statistics of the boundary
        # datastore, shared by all models trained on it (the same text is used
        # in artifact.yaml below)
        if not is_placeholder:
            zarr.open_group(
                str(out / "stats" / f"{BOUNDARY_DATASTORE_NAME}.stats.zarr")
            ).attrs["description"] = BOUNDARY_STATS_DESCRIPTION
            zarr.consolidate_metadata(
                str(out / "stats" / f"{BOUNDARY_DATASTORE_NAME}.stats.zarr")
            )

        # created after the configs, as neural-lam warns if the zarr is older
        # than its config
        fp_boundary_config = (
            out / "configs" / f"{BOUNDARY_DATASTORE_NAME}.yaml"
        )
        _create_stats_datastore(
            ds_boundary_stats,
            fp_boundary_config,
            out / "configs" / f"{BOUNDARY_DATASTORE_NAME}.zarr",
        )
        logger.info("checking the stats datastore loads in neural-lam")
        _check_stats_datastore(fp_boundary_config)

        model = yaml.safe_load((args.configs_dir / "model.yaml").read_text())
        meta = dict(
            artifact_name=args.artifact_name,
            description=(
                "ANNA inference package for the DANRA ML LAM model of "
                "arXiv:2504.09340 (checkpoint: Zenodo "
                f"{model['checkpoint']['zenodo_record']}), see "
                f"{REPO_URL} (artifact/README.md)"
            ),
            model_checkpoint=model["checkpoint"],
            assembled_from=dict(
                gefion_1_zip=args.gefion_1_zip.name,
                gefion_1_zip_sha256=_sha256(args.gefion_1_zip),
                boundary_stats=args.boundary_stats.name,
                boundary_stats_placeholder=is_placeholder,
                boundary_stats_attrs={
                    k: str(v) for k, v in ds_boundary_stats.attrs.items()
                }
                | (
                    {}
                    if is_placeholder
                    else {"description": BOUNDARY_STATS_DESCRIPTION}
                ),
                repo=REPO_URL,
                repo_git=_git_describe(),
            ),
            checkpoint_included=False,
            created_on=datetime.datetime.now()
            .replace(microsecond=0)
            .isoformat(),
        )
        (out / "artifact.yaml").write_text(
            yaml.safe_dump(meta, sort_keys=False)
        )
        shutil.copy(
            REPO_DIR / "artifact" / "package-README.md", out / "README.md"
        )

        layout.check_artifact(out)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), str(args.output))
    logger.info(f"assembled artifact in {args.output}")

    if args.zip is not None:
        layout.write_zip(args.output, args.zip)
        logger.info(f"packaged artifact as {args.zip}")

    if args.fetch_checkpoint:
        fetch_checkpoint.fetch_checkpoint(args.output)


if __name__ == "__main__":
    main()
