# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.0]

First release of the operational inference container for ANNA, the DANRA
machine-learning limited area model (Adamov et al. 2025,
[arXiv:2504.09340](https://arxiv.org/abs/2504.09340)). It runs a forecast
from DINI initial states with a DINI or IFS boundary, using the
training-only inference artifact (configs, statistics, grids and checkpoint)
published on Zenodo
([10.5281/zenodo.23293148](https://doi.org/10.5281/zenodo.23293148)).
[\#2](https://github.com/dmidk/mlwp-inference-container-anna/pull/2),
@leifdenby

### Added

- The forecast pipeline, `python -m anna_inference`: regrid DINI onto the
  DANRA grid, create the inference datastores (DINI or IFS boundary), build
  the graph, run the neural-lam forecast, convert the output to zarr with CF
  projection metadata and optionally upload it to S3. `--skip-completed`
  reruns only the stages not yet done.
- Configuration by command-line options or `ANNA_*` environment variables
  (every `--some-option` is also `ANNA_SOME_OPTION`), through the generic
  `anna_inference.envargs` module. The S3 input (DINI, IFS) and output each
  have their own endpoint URL and AWS profile.
- Tools to build and publish the inference artifact
  (`anna_inference.artifact`: collect, assemble, ERA5 boundary statistics,
  Zenodo draft).
- The container image (`Containerfile`, built from the Zenodo package) and
  the `scripts/build_image.sh` and `scripts/run_inference_container.sh`
  scripts.
- CI: pytest, pre-commit, the image build on every push and pull request,
  and publishing to `ghcr.io/dmidk/mlwp-inference-container-anna` for `v*`
  tags.
- The package version comes from the git tag (setuptools-scm, `v*` tags
  only) and is available as `anna_inference.__version__`.
- The Zenodo downloads (inference package and checkpoint) retry on
  server and network errors, then fall back to the copies in the GitHub
  release [artifact-anna-danra-2026-10-11](https://github.com/dmidk/mlwp-inference-container-anna/releases/tag/artifact-anna-danra-2026-10-11).

[v0.1.0]: https://github.com/dmidk/mlwp-inference-container-anna/releases/tag/v0.1.0
