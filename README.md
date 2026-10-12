# mlwp-inference-container-anna

This repository builds the inference container images for **ANNA**, which
runs the **DANRA machine-learning limited area model** of Adamov et al.
(2025), ["Building Machine Learning Limited Area Models: Kilometer-Scale
Weather Forecasting in Realistic Settings"](https://arxiv.org/abs/2504.09340)
operationally from **DINI** (interior initial states), with **DINI or IFS** on
the boundary. The model is a hierarchical graph-based model (neural-lam
`hi_lam`, graph `7deg_rect_hi4`) trained on DANRA (2.5 km, 3-hourly) with ERA5
boundary forcing on a 7.19° ring around the DANRA domain.

It holds everything needed for that, in one Python project
([pyproject.toml](pyproject.toml), exact environment in `uv.lock`):
- the inference pipeline (`src/anna_inference/`, run by
  `python -m anna_inference`, the image's entrypoint);
- the [Containerfile](Containerfile) and
  [scripts/build_image.sh](scripts/build_image.sh);
- the tools that build the model's *inference artifact* (configs, training
  statistics, grids, see below; `src/anna_inference/artifact/`, docs in
  [artifact/](artifact/README.md)).

ANNA was developed in
[mlwm-deployment](https://github.com/leifdenby/mlwm-deployment)
(`configurations/ANNA`) and moved here with its git history. The full
history, decisions and the reasons behind them are in
[HISTORY.md](HISTORY.md). Paths in it refer to the layout in mlwm-deployment
(e.g. `configurations/ANNA/src/regrid_dini.py` is now
`src/anna_inference/regrid_dini.py`, `inference-artifact/` is now
`artifact/` and `src/anna_inference/artifact/`).

## Model

Described in [artifact/configs/model.yaml](artifact/configs/model.yaml)
(checkpoint, graph recipe, neural-lam arguments, training neural-lam config),
which is part of the inference artifact and is read by the entrypoint and the
container build.

- Checkpoint: the paper's DANRA model, Zenodo
  [10.5281/zenodo.15131838](https://doi.org/10.5281/zenodo.15131838)
  (`danra_model.ckpt`, CC-BY-4.0, by S. Adamov, J. Oskarsson and K. S. Hintz)
- Training: wandb [`jo-research-team/neural_lam/hfzfhiha`](https://wandb.ai/jo-research-team/neural_lam/runs/hfzfhiha)
  (`train-hi_lam-2x300-02_17_19-2867`), a 3-epoch fine-tune with 4-step
  rollouts of `train-hi_lam-2x300-02_13_14-1703`, with
  [`joeloskarsson/neural-lam-dev`](https://github.com/joeloskarsson/neural-lam-dev)
  commit `e7d11c9`
- Run without `--dynamic_time_deltas`: the model was trained before that flag
  existed, and leaving it off reproduces its behaviour

An earlier artifact, `gefion-1`, held an unpublished ablation run (graph
`7deg_rect_hi3`, trained with `--dynamic_time_deltas`), not the paper's model.
It was trained on the same datastores, so it still supplies the training
statistics and grids. See
[artifact/README.md](artifact/README.md#gefion-1-a-prior-incorrect-inference-artifact).

## Software

Pinned in [pyproject.toml](pyproject.toml):

- **neural-lam**: `joeloskarsson/neural-lam-dev@a44d432e` (`research` branch,
  contains the training code and forecast-format boundary forcing, needed for
  IFS; later commits break installing from git)
- **mllam-data-prep**: `sadamov/mllam-data-prep@dd9af481`, the version used
  for training. mllam-data-prep ≥ 0.7 doesn't keep per-point lat/lon for
  regular lat/lon grids, which neural-lam-dev needs for the boundary
- **zarr 2**, as in training; neural-lam-dev writes its output with a zarr v2
  compressor

## Configs

The configs are split between what the model **is** and how this deployment
**feeds** it.

**The trained model:** part of the inference artifact, kept in this
repository in [artifact/configs/](artifact/configs/) as inputs to building it.

| File | What |
|---|---|
| `model.yaml` | the model: checkpoint (Zenodo URL, md5), graph recipe, neural-lam arguments, and the training neural-lam config |
| `7deg_config_era5.yaml` | the neural-lam config the model was trained with |
| `danra_model1_config.yaml` | interior (DANRA) datastore, as in training; the template for the inference interior datastore |
| `era_7deg_model1_config.yaml` | ERA5 boundary datastore the model was trained with; the reference for the boundary features and their (ERA5) normalisation statistics |

**Operational boundaries:** in the package,
[src/anna_inference/boundary/](src/anna_inference/boundary/), and so in the
image. The model wasn't trained on these; they produce the ERA5 boundary
features from other forecasts.

| File | What |
|---|---|
| `ifs_7deg_model1_config.yaml` | operational IFS boundary. **Its header is the contract for the IFS GRIB → zarr conversion**: variables, units, dims, lead times, and the 0.25° box lat 40.0–72.0, lon −27.0–39.5 (the east edge must be exactly 39.5°E) |
| `dini_7deg_model1_config.yaml` | DINI boundary, in the same layout as IFS |

For a forecast, `anna_inference.model_configs.inference_neural_lam_config`
derives the neural-lam config from the training one. It keeps the same model
and settings, with:
- the inference interior datastore;
- the operational boundary datastore, normalised with the ERA5 training
  statistics (`overload_stats_path`).

`tests/test_boundary_configs.py` checks that the operational boundaries give
exactly the ERA5 boundary features, in the order the checkpoint expects.

## Inference artifact

An **inference artifact** is a directory (shipped as a zip) that holds
everything needed to run a trained model's forecasts, apart from the input
data and the software (the term comes from
[mlwm-deployment](https://github.com/leifdenby/mlwm-deployment)). It
normally has:
- the model checkpoint (weights);
- the configs the model was trained with (neural-lam and mllam-data-prep
  datastore configs) and its training arguments;
- the training dataset statistics, which inputs are normalised with and
  outputs de-normalised with.

Without the artifact, the checkpoint alone can't be run. The model expects
its inputs in a particular feature order and normalisation, and on a
particular grid and graph, and these only follow from the configs and
statistics it was trained with.

The artifact should be collected on the machine the model was trained on,
where the training datastores are. ANNA's had to be reconstructed after the
fact: the boundary configs, statistics and grid were missing (see
[artifact/README.md](artifact/README.md)).

The artifact describes the model **as trained** (DANRA, ERA5) and nothing
else. The operational boundaries (DINI, IFS) aren't part of it; they are
part of this repository's code. ANNA's artifact has:

| Path | What |
|---|---|
| `configs/` | the trained-model configs above (`model.yaml`, the training neural-lam config, the DANRA and ERA5 datastore configs) |
| `configs/era_7deg_model1_config.zarr` | the ERA5 boundary statistics in the form neural-lam reads them (`overload_stats_path`) |
| `stats/` | training statistics of the DANRA interior and the ERA5 boundary datastores (the ERA5 ones recomputed exactly) |
| `grids/` | the DANRA grid and statics (target grid for regridding DINI) and the 18,014 ERA5 boundary points the model was trained with |
| `artifact.yaml`, `README.md` | provenance and description |
| `danra_model.ckpt` | the checkpoint |

The checkpoint is already public (Zenodo 15131838), so it isn't re-published.
The artifact is therefore published **without the checkpoint**, as the
*inference package* zip on Zenodo: record
[23293148](https://zenodo.org/records/23293148) (DOI
`10.5281/zenodo.23293148`, `anna-danra-2026-10-11.zip`). As of 2026-10-11
this is a **draft**, which can't be downloaded until it's published.

The container build puts the two back together:
- it fetches the package, or uses a locally built one;
- it downloads the checkpoint into it (md5-checked);
- the result is the complete artifact at `/workspace/inference_artifact`,
  the entrypoint's default `--artifact-path`.

How the artifact is built, checked and published, and its provenance, are in
[artifact/README.md](artifact/README.md). That includes the superseded
`gefion-1` artifact, which held a different model. The README shipped inside
the package is [artifact/package-README.md](artifact/package-README.md).

## Building the image

```bash
CONTAINER_APP=podman scripts/build_image.sh
```

The image is tagged `mlwp-inference-container-anna:latest` (`IMAGE_NAME`).

- `ARTIFACT_SOURCE=zenodo` (default) downloads the published inference package
  (`PACKAGE_URL`/`PACKAGE_MD5`, default Zenodo record 23293148). The script
  checks first that it can be downloaded, i.e. that the record is published.
- `ARTIFACT_SOURCE=local` uses a locally built artifact directory,
  `LOCAL_ARTIFACT_DIR` (default `artifact/build/anna-local`).

In both cases the model checkpoint is downloaded from Zenodo during the build
(md5-checked) if it isn't already there. No AWS credentials are needed. Set
`PULL_PROXY` to pull the base image through DMI's proxy.

CI ([container-image.yml](.github/workflows/container-image.yml)) builds
the x86_64 image from the Zenodo package on every push and pull request.
Tags `v*` pushed to the dmidk repository also publish it as
`ghcr.io/dmidk/mlwp-inference-container-anna:<tag>` and `:latest`
(forks only build).

## Running a forecast

`python -m anna_inference` (also installed as `anna-inference`, the image's
entrypoint, [src/anna_inference/run.py](src/anna_inference/run.py)) runs
the whole pipeline, each step as a subprocess:

1. `anna_inference.regrid_dini`: DINI → DANRA grid (interior) and, for the
   DINI boundary, → the 0.25° ERA5 boundary box. Reads DINI concurrently
   with dask
2. `anna_inference.create_inference_dataset`: inference datastores and
   neural-lam config
3. `neural_lam.build_rectangular_graph`: the model's graph (`model.yaml`)
4. `neural_lam.train_model --eval test`: the forecast
5. `anna_inference.convert_output`: DANRA-like output datasets
6. `anna_inference.upload_outputs`: optionally, the upload to S3

### Configuration

Every command line option `--some-option` can also be set with the
environment variable `ANNA_SOME_OPTION`. An option on the command line wins
over its variable, which wins over the default:

| Option | Environment variable | Default | |
|---|---|---|---|
| `--analysis-time` | `ANNA_ANALYSIS_TIME` | (required) | DINI analysis time, ISO8601, e.g. `2026-10-01T00:00Z` (UTC if no timezone); 00/06/12/18 UTC with the DINI boundary |
| `--forecast-duration` | `ANNA_FORECAST_DURATION` | `PT18H` | multiple of 3 h, between 12 h (neural-lam needs that to load the model) and DINI forecast length − 6 h (30 h for a 36 h DINI run) |
| `--boundary-source` | `ANNA_BOUNDARY_SOURCE` | `dini` | `dini` or `ifs` |
| `--dini-root` | `ANNA_DINI_ROOT` | `s3://harmonie-zarr/dini/control/{analysis}/` | DINI forecast zarrs |
| `--ifs-boundary-path` | `ANNA_IFS_BOUNDARY_PATH` | | IFS forecast zarr, required for `--boundary-source ifs` |
| `--workdir` | `ANNA_WORKDIR` | `inference_workdir` | working directory, with the outputs in `outputs/` |
| `--artifact-path` | `ANNA_ARTIFACT_PATH` | `inference_artifact` | the inference artifact (with checkpoint); outside the container e.g. `artifact/build/<name>` |
| `--skip-completed` | `ANNA_SKIP_COMPLETED` | `false` | skip the stages already completed in the workdir (e.g. rerun only the forecast) |
| `--output-s3-prefix` | `ANNA_OUTPUT_S3_PREFIX` | | if set (e.g. `s3://anna-forecasts/`), the outputs are also uploaded to `{prefix}/<analysis time>/` (e.g. `2026-10-01T000000Z/`), replacing an earlier upload for that analysis time, and checked after uploading |
| `--input-s3-endpoint-url` | `ANNA_INPUT_S3_ENDPOINT_URL` | (AWS config) | S3 endpoint for reading DINI and IFS |
| `--input-aws-profile` | `ANNA_INPUT_AWS_PROFILE` | (AWS config) | AWS profile for reading DINI and IFS |
| `--output-s3-endpoint-url` | `ANNA_OUTPUT_S3_ENDPOINT_URL` | (AWS config) | S3 endpoint for the upload (needs `--output-s3-prefix`) |
| `--output-aws-profile` | `ANNA_OUTPUT_AWS_PROFILE` | (AWS config) | AWS profile for the upload (needs `--output-s3-prefix`) |
| `--debugger` | `ANNA_DEBUGGER` | | `ipdb` to debug the python steps on exceptions |
| `--log-level` | `ANNA_LOG_LEVEL` | `INFO` | loguru level |

- Flags take `true`/`false` (also `1`/`0`, `yes`/`no`) from their variable.
- An empty variable counts as unset.
- Variables in a `.env` file in the current directory are loaded too
  (e.g. S3 credentials); variables already set win.
- A set `ANNA_*` variable that matches no option is an error, so a
  misspelling can't go unnoticed:
  `unknown environment variable(s) ANNA_FORECAST_DURATON (did you mean ANNA_FORECAST_DURATION?)`.
- Configuration errors exit with code 2, a failed step with 1.
- The run starts by logging the version and the effective configuration:
  every option with its variable name, value and where the value came from
  (`cli`, `env` or `default`).

The mapping between options and variables is done by
[src/anna_inference/envargs.py](src/anna_inference/envargs.py). It has
nothing ANNA-specific and only uses the standard library, so other MLWP
inference containers can use the same convention with their own prefix.

### S3 endpoints and credentials

The input (DINI, and an IFS zarr on S3) and the output (the upload with
`--output-s3-prefix`) are configured separately, so they can be on
different S3 hosts with different credentials:
- `--{input,output}-s3-endpoint-url` sets the endpoint for that side's steps
  (as botocore's `AWS_ENDPOINT_URL_S3`).
- `--{input,output}-aws-profile` sets the profile (from `~/.aws`, which
  `run_inference_container.sh` mounts read-only). It also takes precedence
  over `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_SESSION_TOKEN` in the
  environment, which are dropped for that side.
- Unset, a side uses the usual AWS configuration: `AWS_PROFILE`, `AWS_*`
  keys, `AWS_ENDPOINT_URL`, or an `endpoint_url` in the profile in
  `~/.aws/config`.
- For a bucket on AWS, use the region's endpoint (DINI's `harmonie-zarr`:
  `https://s3.eu-central-1.amazonaws.com`), or leave the endpoint unset. With
  the global `https://s3.amazonaws.com` set explicitly, botocore doesn't
  follow the redirect to the bucket's region, and reading fails with
  "group not found".

For example, reading DINI with one profile and uploading to another host:

```bash
ANNA_INPUT_AWS_PROFILE=dini-reader \
ANNA_OUTPUT_S3_PREFIX=s3://anna-forecasts/ \
ANNA_OUTPUT_S3_ENDPOINT_URL=https://s3.example.org \
ANNA_OUTPUT_AWS_PROFILE=forecast-writer \
    uv run python -m anna_inference --analysis-time 2026-10-01T00:00Z
```

### Running it

Outside the container, run it in the uv environment of this repository
(`uv sync`), with AWS credentials for reading DINI (e.g. `AWS_PROFILE`):

```bash
uv run python -m anna_inference --artifact-path artifact/build/anna-local \
    --analysis-time 2026-10-01T00:00Z --forecast-duration PT18H
# or, equivalently
ANNA_ARTIFACT_PATH=artifact/build/anna-local \
    ANNA_ANALYSIS_TIME=2026-10-01T00:00Z ANNA_FORECAST_DURATION=PT18H \
    uv run python -m anna_inference
```

A 12 h forecast takes about 5 minutes on a DGX Spark (GB10 GPU), and needs
about 28 GB of GPU memory and up to about 6 GB of RAM (regridding, graph).

In the container, with [scripts/run_inference_container.sh](scripts/run_inference_container.sh):

```bash
CONTAINER_APP=docker AWS_PROFILE=... \
    scripts/run_inference_container.sh 2026-10-01T00:00Z PT12H
```

- It passes every set `ANNA_*` variable to the container, and mounts the
  host directory `ANNA_WORKDIR` (default `./inference_workdir`) as the
  container's workdir, for the inputs and outputs.
- AWS credentials are passed at run time (`AWS_*` variables and `~/.aws`,
  read-only), never built into the image.
- With docker it runs as the calling user, so the outputs are yours.

Or directly, with options as arguments after the image name or as `-e`
variables:

```bash
docker run --rm --gpus all --shm-size=32g \
    -v ~/.aws:/root/.aws:ro -e AWS_PROFILE=... \
    -v $PWD/inference_workdir:/workspace/inference_workdir \
    -e ANNA_ANALYSIS_TIME=2026-10-01T00:00Z \
    -e ANNA_OUTPUT_S3_PREFIX=s3://anna-forecasts/ \
    mlwp-inference-container-anna:latest --forecast-duration PT12H
```

The image gives the same forecast as the uv environment: identical outputs
for DINI 2026-10-01T00Z PT12H, with the image's NVIDIA torch 2.10.

### Outputs and conventions

`<workdir>/outputs/single_levels.zarr` (pres_seasurface, t2m, u10m,
v10m, pres0m, lwavr0m, swavr0m) and `pressure_levels.zarr` (z, t, r, u, v, tw
at 100–1000 hPa) are on the DANRA grid at valid times T+6 h …
T+`FORECAST_DURATION` in 3 h steps. The initial states are DINI at T+0 and
T+3 h. The conventions are DANRA's:

- **winds are relative to the DANRA (Lambert) grid**, not east/north
- relative humidity `r` is a fraction (0–1)
- `lwavr0m`/`swavr0m` are net surface radiation fluxes
- `tw` is geometric vertical velocity (m/s)

The grid is described with CF metadata as in the DINI zarrs: `x`/`y` in
metres of the DANRA Lambert projection, 2D `lat`/`lon`, and a
`danra_projection` grid mapping variable named by every field's
`grid_mapping`. Viewers such as Gridlook need it to recognise the grid.

## Known limitations

- **DINI boundary.** 25% of the model's boundary points (4,573 of 18,014) lie
  outside the DINI domain, mostly to the east and north. They are filled with
  the value of the nearest DINI grid point. IFS covers the whole ring and is
  the preferred boundary source once the IFS zarr is available.

  ![Boundary points vs the DINI domain](docs/anna_boundary_vs_dini.png)

- **DINI boundary: 00/06/12/18 UTC only.** The model was trained with a
  6-hourly ERA5 boundary at 00/06/12/18 UTC. The boundary is therefore given
  to neural-lam as valid times on that grid, which for each prediction uses
  the last boundary time at or before it ± 6 h. A forecast from, say, 03Z
  needs the boundary at 00Z, which its own DINI run doesn't have. IFS
  boundaries can come from an older cycle (the latest at or before the
  analysis time).

- **DINI zarr retention** is two weeks. Keep regridded development cases
  locally in `dev-data/` (gitignored).

## Development

```bash
uv sync                          # the environment (exact versions in uv.lock)
uv run pre-commit install        # black (79), isort, flake8 on every commit
uv run pytest tests
```

- [artifact/](artifact/README.md): building, checking and publishing the
  inference artifact
- [dev-utils/](dev-utils/):
  - `check_forecast_output.py`: sanity checks of a forecast, against DINI;
  - `check_inference_sample.py`: checks the sample neural-lam builds from
    the inference datastores;
  - `plot_boundary_coverage.py`: the boundary coverage map above.
- Each pipeline step can also be run on its own, e.g.
  `uv run python -m anna_inference.regrid_dini --help`.

## Licence

The code is licensed under the [MIT licence](LICENSE). The published
inference package (configs, statistics, grids) is CC-BY-4.0, and the model
checkpoint (Zenodo 15131838) is CC-BY-4.0 by its authors.
