FROM nvidia/cuda:12.3.1-base-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Prague

RUN apt-get update \
    && apt-get install -y \
    tzdata \
    git \
    python3-pip \
    python-is-python3 \
    build-essential \
    cmake \
    gcc \
    ffmpeg

RUN pip install --upgrade pip

COPY openclip-service/ /root/openclip-service
COPY data/ /root/data

RUN pip install poetry

RUN cd /root/openclip-service \
    && poetry config virtualenvs.create false \
    && poetry install

COPY docker/openclip_service_start.sh /root/openclip_service_start.sh

ENTRYPOINT ["/root/openclip_service_start.sh"]

RUN chmod +x /root/openclip_service_start.sh

ENV NETAPP_PORT=5896

EXPOSE 5896
