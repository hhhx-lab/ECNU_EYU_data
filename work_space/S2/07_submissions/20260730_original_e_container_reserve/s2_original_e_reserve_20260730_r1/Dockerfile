# syntax=docker/dockerfile:1.9

ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG CHECKPOINT_SHA256

LABEL org.opencontainers.image.title="BraTS 2026 S2 original E reserve"
LABEL org.opencontainers.image.description="Experimental unvalidated Task 1 inference container"
LABEL org.brats.artifact-status="experimental_unvalidated"
LABEL org.brats.model="original_E"
LABEL org.brats.checkpoint.sha256="${CHECKPOINT_SHA256}"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/app \
    S2_MODEL_ROOT=/opt/model \
    S2_PREPROCESS_WORKERS=4 \
    S2_EXPORT_WORKERS=4 \
    nnUNet_raw=/tmp/nnunet_raw \
    nnUNet_preprocessed=/tmp/nnunet_preprocessed \
    nnUNet_results=/tmp/nnunet_results

WORKDIR /opt/app

COPY requirements-inference.txt /opt/app/requirements-inference.txt
RUN python -m pip install --no-cache-dir -r /opt/app/requirements-inference.txt \
    && python -m pip check

COPY app/ /opt/app/
COPY model/dataset.json /opt/model/dataset.json
COPY model/plans.json /opt/model/plans.json
RUN mkdir -p /opt/model/fold_0

# The checkpoint is deliberately the last content layer so a later selected
# checkpoint can reuse the complete CUDA, Python, and application layer cache.
COPY --from=checkpoint checkpoint_final.pth /opt/model/fold_0/checkpoint_final.pth
RUN test -n "${CHECKPOINT_SHA256}" \
    && echo "${CHECKPOINT_SHA256}  /opt/model/fold_0/checkpoint_final.pth" | sha256sum -c -

ENTRYPOINT ["python", "/opt/app/run_container.py"]
