# Local Docker laboratory image for the Paper 2 authenticated runtime.
# This is deliberately CPU-only and must never be labelled as a multi-host
# deployment or privacy/security certification.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates openssh-client openssl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
    && python -m pip install --no-cache-dir flwr==1.42.0 numpy==2.5.1

WORKDIR /opt/paper2-app
COPY paper2-dp-secagg-deploy/ /opt/paper2-app/
COPY docker/scripts/ /usr/local/lib/paper2-docker/
RUN test -s /opt/paper2-app/pyproject.toml \
    && test -s /opt/paper2-app/paper2_dp_secagg_deploy/client_app.py \
    && test -s /opt/paper2-app/paper2_dp_secagg_deploy/server_app.py \
    && test -s /usr/local/lib/paper2-docker/bootstrap.sh \
    && test -s /usr/local/lib/paper2-docker/superlink.sh \
    && test -s /usr/local/lib/paper2-docker/node.sh \
    && test -s /usr/local/lib/paper2-docker/coordinator.sh \
    && chmod 0555 /usr/local/lib/paper2-docker/*.sh \
    && python -c "import flwr, numpy, torch; assert flwr.__version__ == '1.42.0'; print(flwr.__version__, numpy.__version__, torch.__version__)"

