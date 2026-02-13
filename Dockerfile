FROM nvidia/cuda:12.9.1-devel-ubuntu22.04 AS base
ENV FORCE_CUDA="1"

# Obtain the UID and GID of the current user to create a user with the same ID, this is to avoid permission issues when mounting local volumes.
ARG USER
ARG UID
ARG GID

# Timezone. Avoid user interaction.
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Add user.
RUN groupadd -g $GID $USER \
    && useradd --uid $UID --gid $GID -m $USER \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER \
    && chmod 0440 /etc/sudoers.d/$USER

# Install Python and some utilities.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python-is-python3 \
    python3-pip \
    python3-setuptools \
    # Graphics libraries.
    libgl1-mesa-glx \
    mesa-utils \
    libglapi-mesa \
    libqt5gui5 \
    fonts-liberation \
    # Utilities.
    curl \
    wget \
    git \
    build-essential \
    ca-certificates \
    cmake \
    jupyter-notebook \
    figlet

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy

# Copy requirements.txt and install Python packages.
USER $USER
WORKDIR /.cache

# Disable jupyter authentication
USER $USER
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.token = ''" >> /home/$USER/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password = ''" >> /home/$USER/.jupyter/jupyter_notebook_config.py

# Change terminal color
ENV TERM=xterm-256color
RUN echo "PS1='\[\e[1;34m\]\u\[\e[1;37m\]@\[\e[1;33m\]\h\[\e[1;37m\]:\[\e[0;37m\]\w\[\e[1;37m\] → \[\e[0m\]'" >> ~/.bashrc
RUN echo "source ~/.bashrc" >> ~/.bash_profile

# Set HF_HOME to a directory inside the workspace to avoid re-downloading models every time.
RUN echo "export HF_HOME="/workspace/.cache/hugging-face"" >> /home/$USER/.bashrc

WORKDIR /workspace
CMD [ "/bin/bash" ]