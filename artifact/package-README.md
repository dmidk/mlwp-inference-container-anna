# ANNA inference package for the DANRA ML LAM model

This package describes the DANRA machine-learning limited area model (LAM) of
Adamov et al. (2025),
["Building Machine Learning Limited Area Models: Kilometer-Scale Weather Forecasting in Realistic Settings"](https://arxiv.org/abs/2504.09340),
**as it was trained**: DANRA on the interior, ERA5 on the boundary. It holds
what is needed to run the model, apart from the checkpoint and the input
data:
- the training configs;
- the training statistics, which inputs are normalised with;
- the grids.

It contains nothing about any particular operational data source. At DMI
("ANNA") the model is run from DINI (interior) and DINI or IFS (boundary)
forecasts. Those boundary datastores, and how they are mapped onto the
training ERA5 boundary, are part of the deployment repository (see "Using
it").

**The model weights are not in this package.** They are the paper's published
DANRA checkpoint, Zenodo
[10.5281/zenodo.15131838](https://doi.org/10.5281/zenodo.15131838)
(`danra_model.ckpt`, CC-BY-4.0). Download them separately (URL and md5 in
`configs/model.yaml`) and put the file next to this README as
`danra_model.ckpt`.

The package's version, creation time and provenance are recorded in
`artifact.yaml`.

## Contents

| Path | What |
|---|---|
| `configs/model.yaml` | the model: checkpoint (Zenodo URL, md5), graph recipe (`7deg_rect_hi4`), neural-lam `train_model` arguments, and the training neural-lam config |
| `configs/7deg_config_era5.yaml` | the neural-lam config the model was trained with |
| `configs/danra_model1_config.yaml` | interior (DANRA) mllam-data-prep datastore config, as in training |
| `configs/era_7deg_model1_config.yaml` | ERA5 boundary datastore config the model was trained with: the reference for the boundary features and their normalisation |
| `configs/era_7deg_model1_config.zarr` | ERA5 boundary "stats datastore" that neural-lam opens through `overload_stats_path`, to normalise a boundary from another source with the ERA5 training statistics (statistics and splits only) |
| `stats/danra_model1_config.stats.zarr` | training statistics of the DANRA interior datastore |
| `stats/era_7deg_model1_config.stats.zarr` | training statistics of the ERA5 boundary datastore, recomputed exactly from WeatherBench2 ERA5 (2000-01-01 to 2018-10-29, 6-hourly) |
| `grids/danra_model1_config.grid.zarr` | DANRA grid (x, y, lat, lon) and static fields (land-sea mask, orography) from DANRA v0.5.0: the grid interior initial states must be on |
| `grids/era_7deg_model1_config.grid.zarr` | lat/lon of the 18,014 ERA5 (0.25°) boundary points the model was trained with, a 7.19° ring around the DANRA domain |
| `artifact.yaml` | version and provenance of this package |

## Using it

The package is made for
[mlwp-inference-container-anna](https://github.com/leifdenby/mlwp-inference-container-anna),
which builds ANNA's inference container images. Its container build downloads
this package and the checkpoint. Its entrypoint (`python -m anna_inference`) then:
- regrids DINI;
- creates the inference datastores, from the training interior config and
  its own DINI/IFS boundary datastore configs;
- derives the neural-lam config from the training one, normalising the
  boundary with the ERA5 training statistics;
- builds the graph, runs the forecast and converts the output.

To run the model from other data, provide the interior on the DANRA grid and
a boundary with the features of `era_7deg_model1_config.yaml`, in the same
way.

Software as used in training:
- neural-lam from
  [joeloskarsson/neural-lam-dev](https://github.com/joeloskarsson/neural-lam-dev)
  (`research` branch);
- mllam-data-prep from `sadamov/mllam-data-prep` (`building-ml-lams` branch);
- zarr 2.

Exact pins are in `pyproject.toml` and `uv.lock` of
mlwp-inference-container-anna.

Conventions of the model's inputs and outputs (those of DANRA):

- **winds are relative to the DANRA (Lambert) grid**, not east/north;
- relative humidity `r` is a fraction (0–1);
- `tw` is geometric vertical velocity (m/s) in the interior. The boundary
  uses pressure vertical velocity ω (Pa/s), as in ERA5.

## Licence and acknowledgements

The package is licensed CC-BY-4.0. The model, its training setup and the
datastore configurations are the work of the authors of arXiv:2504.09340:

> Simon Adamov, Joel Oskarsson, Leif Denby, Tomas Landelius, Kasper Hintz,
> Simon Christiansen, Irene Schicker, Carlos Osuna, Fredrik Lindsten, Oliver
> Fuhrer and Sebastian Schemm

When you use the model, please cite the paper and the checkpoint record
(10.5281/zenodo.15131838).
