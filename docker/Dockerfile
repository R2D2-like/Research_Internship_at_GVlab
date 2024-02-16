# Base image
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Set the default shell to bash for compatibility
SHELL ["/bin/bash", "-c"]

# Install common dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    pkg-config \
    unzip \
    wget \
    git \
    zsh \
    vim \
    python3-pip \
    lsb-release \
    gnupg \
    libsdl2-2.0-0 \
    cmake \
    xvfb \
    xorg-dev \
    freeglut3-dev \
    libglu1-mesa-dev \
    x11-apps \
    libgl1-mesa-dev \
    libglib2.0-0 \
    bash-completion

# Enable bash-completion and configure it to source automatically on container startup
RUN echo 'if [ -f /etc/bash_completion ]; then . /etc/bash_completion; fi' >> /etc/bash.bashrc

# Install dependencies for robomimic and other utilities
RUN apt-get install -y \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    patchelf \
    python-opengl icewm

# Configure zsh with oh-my-zsh
RUN wget http://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
RUN sed -i 's/# DISABLE_AUTO_UPDATE="true"/DISABLE_AUTO_UPDATE="true"/g' ~/.zshrc
RUN echo 'alias vnc="export DISPLAY=:0; Xvfb :0 -screen 0 1400x900x24 &; x11vnc -display :0 -forever -noxdamage > /dev/null 2>&1 &; icewm-session &"' >> ~/.zshrc
RUN echo 'source "$HOME/mambaforge/etc/profile.d/conda.sh"' >> ~/.zshrc
RUN echo 'source "$HOME/mambaforge/etc/profile.d/mamba.sh"' >> ~/.zshrc

# Install Python packages
RUN pip3 install --upgrade pip
RUN pip3 install opencv-python PyYAML

# Setup bash for color prompt
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' ~/.bashrc

# Install Miniforge for conda environment management
COPY conda_environment.yaml /root/
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh" && \
    bash Mambaforge-Linux-x86_64.sh -b && \
    ~/mambaforge/bin/conda init bash && \
    ~/mambaforge/bin/mamba env create -f /root/conda_environment.yaml && \
    echo 'conda activate robodiff' >> ~/.bashrc

# Clone and install robosuite
RUN mkdir -p /root/external/ && cd /root/external/ && git clone https://github.com/R2D2-like/robosuite.git && \
    cd /root/external/robosuite && pip3 install -r requirements.txt && \
    pip3 install -r requirements-extra.txt && python3 /root/external/robosuite/robosuite/scripts/setup_macros.py

# Cleanup
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

# Set default command
CMD ["zsh"]
