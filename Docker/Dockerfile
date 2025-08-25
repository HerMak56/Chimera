# Docker/Dockerfile
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Etc/UTC
ARG USER=oversir
ARG UID=1000
ARG GID=1000
ARG TORCH_CUDA_TAG=cu121  # PyTorch CUDA wheels tag

ENV TZ=${TZ}
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo "${TZ}" > /etc/timezone

# Base packages + GUI bits
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo curl ca-certificates git nano \
    python3 python3-venv python3-distutils python3-pip python-is-python3 \
    build-essential pkg-config \
    ffmpeg \
    libgl1 libgl1-mesa-dri libegl1 \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    python3-tk qtbase5-dev python3-pyqt5 \
    git \
 && rm -rf /var/lib/apt/lists/*

# User
RUN groupadd --gid ${GID} ${USER} \
 && useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash ${USER} \
 && usermod -aG sudo ${USER} \
 && echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# XDG/X11
ENV XDG_RUNTIME_DIR=/tmp/runtime-${USER}
ENV QT_X11_NO_MITSHM=1
RUN mkdir -p ${XDG_RUNTIME_DIR} && chmod 700 ${XDG_RUNTIME_DIR} && chown ${USER}:${USER} ${XDG_RUNTIME_DIR}

# Install uv system-wide
RUN pip install --no-cache-dir uv

# Create venv managed by uv and give it to ${USER}
ENV VENV_PATH=/opt/venv
ENV PATH=${VENV_PATH}/bin:/usr/local/bin:/usr/bin:/bin
RUN uv venv ${VENV_PATH} \
 && chown -R ${USER}:${USER} ${VENV_PATH}

RUN pip3 install --upgrade pip && \
    git clone https://github.com/egowalk/egowalk-dataset.git /opt/egowalk-dataset && \
    uv pip install -e /opt/egowalk-dataset
# Workdir
WORKDIR /home/${USER}/workspace

# Pre-copy requirements for caching (optional)
COPY requirements.txt /tmp/requirements.txt


# Install deps into the venv via uv pip
RUN uv pip install --upgrade pip setuptools wheel \
 && uv pip install --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG} \
        torch torchvision torchaudio \
 && if [ -f /tmp/requirements.txt ]; then uv pip install -r /tmp/requirements.txt; fi \
 && uv pip install "comet_ml>=3.44.0"

# (Optional) egowalk-dataset as editable
# RUN git clone https://github.com/egowalk/egowalk-dataset.git /opt/egowalk-dataset \
#  && uv pip install -e /opt/egowalk-dataset

# X11 at runtime
ENV XAUTHORITY=/tmp/.Xauthority

# Home ownership
RUN chown -R ${USER}:${USER} /home/${USER}

USER ${USER}
CMD ["/bin/bash"]