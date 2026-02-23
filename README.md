# ANNA

The ANNA artifact is "s3://mlwm-artifacts/inference-artifacts/gefion-1.zip", which contains a model trained on the DANRA dataset on Gefion.

## Building image
`AWS_ACCESS_KEY_ID=<access_key> AWS_SECRET_ACCESS_KEY=<secret_access_key> CONTAINER_APP=podman ./build_image.sh`

## Running inference
`AWS_ACCESS_KEY_ID=<access_key> AWS_SECRET_ACCESS_KEY=<secret_access_key> ./run_inference_container.sh 2026-02-04T00:00:00Z`
