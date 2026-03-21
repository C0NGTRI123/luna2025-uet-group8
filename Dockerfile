FROM --platform=linux/amd64 pytorch/pytorch AS example-algorithm-amd64
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Copy the lung-nodule package and requirements
COPY --chown=user:user lung-nodule /opt/app/lung-nodule
COPY --chown=user:user requirements-ai.txt /opt/app/

# Install dependencies and the lung-nodule package
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements-ai.txt \
    /opt/app/lung-nodule[detection]

# Copy weight files (data, not code)
COPY --chown=user:user results /opt/app/resources

# Copy the thin GC entrypoint
COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
