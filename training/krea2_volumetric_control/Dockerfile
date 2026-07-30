FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG UV_VERSION=0.8.4
ARG UPSTREAM_TRAINER_COMMIT=909682ae0bdd9eb87c8258894c0003224db00d0b

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh

WORKDIR /opt/k2lab
COPY . /opt/k2lab
RUN /root/.local/bin/uv sync \
    --project /opt/k2lab/training/krea2_volumetric_control \
    --frozen \
    --extra dev
RUN git clone https://github.com/Tanmaypatil123/Krea-2-controlnet.git /opt/Krea-2-controlnet \
    && git -C /opt/Krea-2-controlnet checkout "${UPSTREAM_TRAINER_COMMIT}" \
    && test "$(git -C /opt/Krea-2-controlnet rev-parse HEAD)" = "${UPSTREAM_TRAINER_COMMIT}"

WORKDIR /opt/k2lab/training/krea2_volumetric_control
ENTRYPOINT ["/opt/k2lab/training/krea2_volumetric_control/.venv/bin/python"]
CMD ["train.py", "--help"]
