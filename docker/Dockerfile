# Base image
FROM ubuntu:20.04

# Set the default shell to bash for compatibility
SHELL ["/bin/bash", "-c"]

#timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

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
    # xeyes
    x11-apps \
    # glxgears
    mesa-utils  \
    libgl1-mesa-dev \
    libglib2.0-0 \
    bash-completion

# Enable bash-completion and configure it to source automatically on container startup
RUN echo 'if [ -f /etc/bash_completion ]; then . /etc/bash_completion; fi' >> /etc/bash.bashrc

# not to be asked keyboard-configuration
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y  -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" keyboard-configuration

# Install dependencies for robomimic and other utilities
RUN apt-get update && apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    libglfw3-dev \
    patchelf

# Install Python packages
RUN pip3 install --upgrade pip
RUN pip3 install opencv-python PyYAML

# Setup bash for color prompt
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' ~/.bashrc

# Clone and install robosuite
RUN mkdir -p /root/external/ && cd /root/external/ && git clone -b ri@gvlab https://github.com/R2D2-like/robomimic.git && \
    cd /root/external/robomimic && pip install -e .

RUN pip3 install mujoco

# uninstall 'blinker'
RUN apt-get remove python3-blinker -y
RUN pip3 uninstall PyOpenGL PyOpenGL_accelerate -y
# make sure the version of PyOpenGL is 3.1.7
RUN pip3 install PyOpenGL==3.1.7 PyOpenGL_accelerate==3.1.7
RUN cd /root/external/ && git clone -b ri@gvlab https://github.com/R2D2-like/robosuite.git \
    && cd /root/external/robosuite && pip3 install -r requirements.txt && \
    pip3 install -r requirements-extra.txt
RUN python3 /root/external/robosuite/robosuite/scripts/setup_macros.py
# RUN python3 /usr/local/lib/python3.10/dist-packages/robosuite/scripts/setup_macros.py

# set up robosuite
RUN cd /root/external/robosuite && pip3 install -e .

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

ENTRYPOINT []

CMD ["/bin/bash"]