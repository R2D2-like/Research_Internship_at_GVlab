FROM ubuntu:20.04
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        wget \
        curl \
        git \
        vim \
        python3-pip \
        lsb-release \
        gnupg 


COPY ./. /root/Research_Internship_at_GVlab/
RUN chmod -R +x /root/Research_Internship_at_GVlab/*

# # Force color prompt in terminal.
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc


# CUDA 11.7
# RUN pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

# xeyes
RUN apt-get update -y; apt-get install -y x11-apps;

# opencv
RUN pip3 install opencv-python
RUN apt-get update && \
    apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libglib2.0-0

# yaml
RUN pip3 install PyYAML

# pip
RUN pip3 install --upgrade pip

WORKDIR /root
RUN mkdir -p external/ && cd external/ && git clone https://github.com/ARISE-Initiative/robosuite.git \
    && cd /root/external/robosuite && pip3 install -r requirements.txt && python3 /root/external/robosuite/robosuite/scripts/setup_macros.py
WORKDIR /root

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

ENTRYPOINT []

CMD ["/bin/bash"]