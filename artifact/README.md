# Building the ANNA inference package

This is how the **inference package** is built: everything ANNA needs to run
the paper's DANRA model except the checkpoint (configs, training statistics,
grids and the model description). Once built, the package is published on
Zenodo, and the container build (`ARTIFACT_SOURCE=zenodo`) downloads it along
with the checkpoint.

- [package-README.md](package-README.md) is the README shipped inside the
  package, describing its contents to users of the Zenodo record.
- Built packages go to `artifact/build/` (gitignored):
  `artifact/build/<artifact-name>/` and, with `--zip`,
  `artifact/build/<artifact-name>.zip`.

The code is in the `anna_inference.artifact` package
(`src/anna_inference/artifact/`); run each module with
`uv run python -m anna_inference.artifact.<module>` from the repository root.

| Module / file | What |
|---|---|
| `assemble` | builds the package from the inputs below |
| `era5_boundary_stats` | exact ERA5 boundary training statistics from WeatherBench2 |
| `check_checkpoint_compat` | strict checkpoint load and one-step forecast on synthetic data with the package's configs |
| `zenodo_draft` | uploads the package zip as a Zenodo **draft** (never publishes) |
| `sanitize_checkpoint` | loads/cleans the gefion-1 checkpoint (see [gefion-1](#gefion-1-a-prior-incorrect-inference-artifact)) |
| `artifact/provenance/` (scripts) | tools used to collect what the package records (below) |

## The model

The paper's DANRA model (arXiv:2504.09340) is described in
[`../configs/model.yaml`](../configs/model.yaml):

- **Checkpoint:** Zenodo
  [10.5281/zenodo.15131838](https://doi.org/10.5281/zenodo.15131838),
  `danra_model.ckpt`, md5 `9b2eaa42fcb0965e450ff27addf9138d`, CC-BY-4.0. It is
  not re-published.
- **Training run:** wandb `jo-research-team/neural_lam/hfzfhiha`
  (`train-hi_lam-2x300-02_17_19-2867`). It is a 3-epoch fine-tune with 4-step
  rollouts of `train-hi_lam-2x300-02_13_14-1703`, at neural-lam-dev commit
  `e7d11c9`. The run is identified by the checkpoint's `ModelCheckpoint`
  callback paths. Its arguments come from `provenance/wandb_run_args.py`.
- **Graph:** `7deg_rect_hi4`. Model arguments as in `model.yaml`, **without**
  `--dynamic_time_deltas`: the model was trained before that flag existed,
  and leaving it off reproduces its behaviour.
- **Datastores:** the same as gefion-1 (`7deg_config.yaml`: DANRA interior and
  ERA5 boundary), so the training statistics and the boundary grid are shared.

## Inputs

1. **`gefion-1.zip`** (`s3://mlwm-artifacts/inference-artifacts/gefion-1.zip`,
   sha256 `d1e581308f272ecb93325318780c3ece253278b8d77a0c9455efe9e78c0771e5`).
   It is the only surviving copy of:
   - the DANRA interior training statistics;
   - the training boundary datastore, pickled in its checkpoint. Its
     `grid_index` gives the exact 18,014 boundary points (stacked
     `[longitude, latitude]` over the 187 × 267 ERA5 subset box).
2. **ERA5 boundary statistics** `era_7deg_model1_config.stats.zarr`. They were
   never exported from Gefion, so they were recomputed exactly from WeatherBench2
   ERA5 on a server with good bandwidth to GCS:

   ```bash
   uv run --with gcsfs python -m anna_inference.artifact.era5_boundary_stats \
       --output era_7deg_model1_config.stats.zarr \
       --partials-dir era5_stats_partials
   ```

   - The run covers 688/688 blocks: 27,505 6-hourly steps from 2000-01-01 to
     2018-10-29, over 49,929 grid points, reading about 9 TB from WB2.
   - Like the training mllam-data-prep, it computes the stats over the whole
     ERA5 subset box before cropping. Its `diff_std` is over second
     differences, but neural-lam only uses `{forcing,static}__train__{mean,std}`
     for the boundary.
   - It was validated against an mllam-data-prep-built datastore to about
     1e-13.
   - `provenance/check_boundary_stats.py` checks the result.
3. The **DANRA v0.5.0** public store, read by `anna_inference.regrid_dini
   create-danra-grid` for the DANRA grid and statics.
4. The **repo configs** in [`../configs/`](../configs/).

## Build

From the repository root, behind DMI's proxy with `truststore`:

```bash
uv run python -m anna_inference.artifact.assemble \
    --gefion-1-zip gefion-1.zip \
    --boundary-stats era_7deg_model1_config.stats.zarr \
    --artifact-name anna-danra-2026-10-06 --zip
```

Behind a TLS-inspecting proxy, run it with the OS trust store:

```bash
uv run --with truststore python -c \
    "import truststore, runpy, sys; truststore.inject_into_ssl(); sys.argv[0] = 'assemble'; runpy.run_module('anna_inference.artifact.assemble', run_name='__main__')" \
    --gefion-1-zip gefion-1.zip ...
```

- Add `--fetch-checkpoint` to also download `danra_model.ckpt` (md5-checked)
  into the package directory for local runs, e.g. with
  `INFERENCE_ARTIFACT_PATH=artifact/build/<name> scripts/entry.sh` or
  `LOCAL_ARTIFACT_DIR=artifact/build/<name> scripts/build_image.sh`. The
  local container build defaults to `artifact/build/anna-local`.
- The checkpoint is never put in the zip.

## Check

```bash
# zip == directory, no checkpoint, no gefion-1 files, no local paths
uv run python artifact/provenance/check_package.py \
    artifact/build/anna-danra-2026-10-06.zip
# strict checkpoint load (561 tensors) and a one-step forecast with the
# package's configs
uv run python -m anna_inference.artifact.check_checkpoint_compat \
    <path>/danra_model.ckpt \
    artifact/build/anna-danra-2026-10-06/configs \
    <workdir> --run-eval
```

## Publish

`anna_inference.artifact.zenodo_draft` creates (or updates) a **draft**. The person publishing
reviews it in the Zenodo web UI and publishes it there, which is
irreversible.

- The token (scope `deposit:write`) is read from `ZENODO_TOKEN` and never
  stored.
- Use `--dry-run` to print the metadata, and `--sandbox` to rehearse on
  sandbox.zenodo.org.

The record:
- is a dataset under CC-BY-4.0;
- creators: Kasper Hintz and Leif Denby (DMI);
- contributors: the other authors of arXiv:2504.09340, who are acknowledged;
- related identifiers: Zenodo 15131838 (`requires`), arXiv:2504.09340,
  neural-lam-dev and this repository.

The author list came from `provenance/arxiv_meta.py 2504.09340` and the
checkpoint record's metadata from `provenance/zenodo_record.py 15131838`.

```bash
ZENODO_TOKEN=... uv run python -m anna_inference.artifact.zenodo_draft \
    --zip artifact/build/anna-danra-2026-10-06.zip
```

After publishing, set `PACKAGE_URL`/`PACKAGE_MD5` in
[`../scripts/build_image.sh`](../scripts/build_image.sh) (and mention the
record in [`../README.md`](../README.md)).

## gefion-1: a prior, incorrect inference artifact

`gefion-1.zip` was the first ANNA inference artifact. It was assembled on
Gefion before this deployment, and this work started out deploying it. It is
**not the paper's model**: `provenance/compare_checkpoints.py` shows
unrelated weights and a different graph. It was replaced on 2026-10-06.

| | gefion-1 | paper model (deployed) |
|---|---|---|
| run | `train-hi_lam-2x300-02_27_15-4034` (wandb `n0o7jw5f`), an unpublished ablation | `hfzfhiha`, evaluated in the paper |
| training | 80 epochs from scratch, 1-step rollouts, neural-lam-dev `e58e334c` | fine-tune with 4-step rollouts, `e7d11c9` |
| graph | `7deg_rect_hi3` (417 tensors) | `7deg_rect_hi4` (561 tensors) |
| `--dynamic_time_deltas` | on | off |
| checkpoint | full, with pickled datastores pointing at Gefion storage (needs `sanitize_checkpoint` to load) | weights only |
| boundary stats / grid | missing from the artifact | (package: exact stats, recovered grid) |

Other problems with gefion-1:
- it shipped no boundary datastore config, statistics or grid;
- an interim local build used placeholder boundary statistics from 3 days of
  ERA5, up to about 1 std off.

It is kept only as an **input** to `anna_inference.artifact.assemble`,
because the paper model was trained on the same datastores. Nothing from
gefion-1 apart from the interior statistics and the boundary grid goes into
the package. `artifact.yaml` records its sha256. The full history is in
[`../HISTORY.md`](../HISTORY.md).
