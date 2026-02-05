#!/usr/bin/env bash
# This script is used to run the inference for the ANNA model.
#
# This script is intended to be run in a container, and assumes that during the
# container image build that the inference artifact was unpacked to
# inference_artifact/. You can also run this script interactively if you have
# extracted the inference artifact yourself.
#
# The selection of datasets to use for input to the model, analysis time and
# forecast duration is controller by the following environment variables:
# DATASTORE_INPUT_PATHS, ANALYSIS_TIME, FORECAST_DURATION and NUM_EVAL_STEPS
# (the latter should be inferred from FORECAST_DURATION, but that is TODO)
#
# - DATASTORE_INPUT_PATHS is a comma-separated list of mappings of
#   {datastore_name}.{input_name}={input_path}
# - ANALYSIS_TIME is the analysis time to start the forecast from is ISO8601
#   format
# - FORECAST_DURATION is the duration of the forecast in ISO8601 duration
#   format and effects the length of the produced inference dataset
# - NUM_EVAL_STEPS is the number of autoregressive steps to run during
#   inference. This should be consistent with FORECAST_DURATION and the model
#   configuration (e.g. if the model was trained on 3-hourly data and
#   FORECAST_DURATION is PT18H then NUM_EVAL_STEPS should be 6

# make this script fail on any error
set -e

## Runtime configuration (variable expected to change on every execution)
# enable use of .env so that during development we can set environment (e.g.
# paths to replace in datastore config)
if [ -f .env ] ; then
    echo "Sourcing local .env file"
    set -a && source .env && set +a
fi

USE_UV=${USE_UV:-true}
if [ "$USE_UV" = true ] ; then
    echo "Using uv to run commands"
    UV_CMD="uv run"
else
    echo "Not using uv to run commands, using plain python"
    UV_CMD=""
fi

# print CUDA debug info
${UV_CMD} python - <<'PY'
import torch, subprocess, os
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print("device capability:", cap)
    print("name:", torch.cuda.get_device_name(0))
    try:
        torch.randn(2, device="cuda")
        print("cuda op: OK")
    except Exception as e:
        print("cuda op failed:", e)
PY
