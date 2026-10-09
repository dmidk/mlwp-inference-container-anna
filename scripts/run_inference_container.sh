#!/bin/bash
# Run an ANNA forecast in the inference image (built with build_image.sh).
#
# Usage:
#   scripts/run_inference_container.sh <ANALYSIS_TIME> [<FORECAST_DURATION>]
#
#   ANALYSIS_TIME      DINI analysis time, ISO8601 (e.g. 2026-10-01T00:00Z)
#   FORECAST_DURATION  ISO8601 duration, PT12H .. PT30H (default PT18H)
#
# Optional environment variables (see scripts/entry.sh for the pipeline):
#   CONTAINER_APP      podman (default) or docker
#   ANNA_IMAGE_NAME    image (default mlwp-inference-container-anna:latest)
#   INFERENCE_WORKDIR  host directory for inputs and outputs, mounted into
#                      the container (default ./inference_workdir); the
#                      forecast is written to outputs/ in it
#   BOUNDARY_SOURCE    dini (default) or ifs
#   IFS_BOUNDARY_PATH  IFS forecast zarr for BOUNDARY_SOURCE=ifs (an s3://
#                      URL, or a local path, which is mounted read-only)
#   DINI_ROOT          DINI forecast zarrs (default: entry.sh's, on S3)
#   SKIP_COMPLETED     true: reuse completed stages in INFERENCE_WORKDIR
#   ANNA_DEBUGGER      ipdb: debug the python steps (interactive)
#
# AWS credentials for reading DINI (and IFS) from S3 are passed at run time,
# never baked into the image: AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY
# (/AWS_SESSION_TOKEN) if set, and ~/.aws (read-only) with AWS_PROFILE.

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ] ; then
    sed -n '4,8p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 1
fi
ANALYSIS_TIME="$1"
FORECAST_DURATION="${2:-PT18H}"

CONTAINER_APP=${CONTAINER_APP:-podman}
ANNA_IMAGE_NAME=${ANNA_IMAGE_NAME:-mlwp-inference-container-anna:latest}
INFERENCE_WORKDIR=${INFERENCE_WORKDIR:-./inference_workdir}
BOUNDARY_SOURCE=${BOUNDARY_SOURCE:-dini}
mkdir -p "${INFERENCE_WORKDIR}"
INFERENCE_WORKDIR="$(cd "${INFERENCE_WORKDIR}" && pwd)"

ARGS=(--rm --shm-size=32g)
if [ "${CONTAINER_APP}" = "docker" ] ; then
    # run as the calling user, so the outputs in INFERENCE_WORKDIR are
    # theirs (rootless podman maps the container's root to the user already).
    # The uid has no passwd entry in the image: USER/LOGNAME are what
    # getpass.getuser() (e.g. torch's compile cache) reads first
    CONTAINER_HOME=/tmp
    ARGS+=(--gpus all --user "$(id -u):$(id -g)" -e HOME="${CONTAINER_HOME}")
    ARGS+=(-e USER="$(id -un)" -e LOGNAME="$(id -un)")
else
    CONTAINER_HOME=/root
    ARGS+=(--device nvidia.com/gpu=all)
fi
if [ "${ANNA_DEBUGGER:-}" = "ipdb" ] ; then
    ARGS+=(-it)
fi

ARGS+=(-v "${INFERENCE_WORKDIR}:/workspace/inference_workdir:Z")
ARGS+=(-e ANALYSIS_TIME="${ANALYSIS_TIME}")
ARGS+=(-e FORECAST_DURATION="${FORECAST_DURATION}")
ARGS+=(-e BOUNDARY_SOURCE="${BOUNDARY_SOURCE}")
ARGS+=(-e INFERENCE_WORKDIR=/workspace/inference_workdir)
for var in DINI_ROOT SKIP_COMPLETED ANNA_DEBUGGER ; do
    if [ -n "${!var:-}" ] ; then
        ARGS+=(-e "${var}=${!var}")
    fi
done

if [ -n "${IFS_BOUNDARY_PATH:-}" ] ; then
    if [ -e "${IFS_BOUNDARY_PATH}" ] ; then
        # a local zarr: mount it at the same path
        IFS_BOUNDARY_PATH="$(cd "${IFS_BOUNDARY_PATH}" && pwd)"
        ARGS+=(-v "${IFS_BOUNDARY_PATH}:${IFS_BOUNDARY_PATH}:ro,Z")
    fi
    ARGS+=(-e IFS_BOUNDARY_PATH="${IFS_BOUNDARY_PATH}")
fi

# AWS credentials, at run time only
for var in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN \
        AWS_PROFILE AWS_REGION AWS_DEFAULT_REGION ; do
    if [ -n "${!var:-}" ] ; then
        ARGS+=(-e "${var}=${!var}")
    fi
done
if [ -d "${HOME}/.aws" ] ; then
    ARGS+=(-v "${HOME}/.aws:${CONTAINER_HOME}/.aws:ro,Z")
fi

echo "Running ${ANNA_IMAGE_NAME}: ${ANALYSIS_TIME} ${FORECAST_DURATION}" \
    "(boundary ${BOUNDARY_SOURCE}), workdir ${INFERENCE_WORKDIR}"
${CONTAINER_APP} run "${ARGS[@]}" "${ANNA_IMAGE_NAME}"
