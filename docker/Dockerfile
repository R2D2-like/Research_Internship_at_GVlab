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

# ROS setting
RUN apt-get update && \
    apt-get install -y \
        avahi-autoipd \
        avahi-daemon \
        avahi-utils

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" | tee /etc/apt/sources.list.d/ros-focal.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt update && \
    apt install -y \
        python3-catkin-pkg-modules \
        python3-rosdistro-modules \
        python3-rospkg-modules
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# # RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt update && \
    apt install -y \
        ros-noetic-desktop-full

RUN apt install -y \
        python3-rosdep \
        python3-rosinstall \
        python3-rosinstall-generator \
        build-essential \
        python3-catkin-tools

RUN rosdep init && rosdep update

# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#         python-rosinstall \
#         python-catkin-tools

RUN mkdir -p /root/catkin_ws/src && \
    cd /root/catkin_ws && \
    bash -c "source /opt/ros/noetic/setup.bash; catkin_make"


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git-core \
        python3-wstool \
        python3-vcstools \
        ros-noetic-control-msgs \
        ros-noetic-xacro \
        ros-noetic-tf2-ros \
        ros-noetic-rviz \
        ros-noetic-cv-bridge \
        ros-noetic-actionlib \
        ros-noetic-actionlib-msgs \
        ros-noetic-dynamic-reconfigure \
        ros-noetic-trajectory-msgs \
        ros-noetic-rospy-message-converter


RUN apt-get update && \
    apt-get install -y \
        ros-noetic-ros-control \
        ros-noetic-control-toolbox \
        ros-noetic-realtime-tools \
        ros-noetic-ros-controllers \
        ros-noetic-xacro \
        ros-noetic-tf-conversions \
        ros-noetic-kdl-parser \
        ros-noetic-moveit-resources-prbt-moveit-config \
        ros-noetic-pilz-industrial-motion-planner 

# set up for UR5e
RUN cd /root/catkin_ws/src && \
    git clone -b ri@gvlab https://github.com/R2D2-like/universal_robot.git

# MoveIt
RUN apt update && \
    apt install ros-noetic-moveit -y

# driver for real robot
# source global ros
RUN /bin/bash -c 'source /opt/ros/noetic/setup.bash'

RUN cd /root/catkin_ws && \
    git clone https://github.com/R2D2-like/Universal_Robots_ROS_Driver.git

ENV ROS_DISTRO noetic

# install universal robot ros packages
RUN apt-get update && apt-get install -y \
    # ROS utils 
    ros-$ROS_DISTRO-gazebo-ros-pkgs \
    ros-$ROS_DISTRO-spacenav-node \
    ros-$ROS_DISTRO-rqt-common-plugins \
    ros-$ROS_DISTRO-rqt-joint-trajectory-controller \
    ros-$ROS_DISTRO-rqt-tf-tree \
    ros-$ROS_DISTRO-rqt-multiplot \
    ros-$ROS_DISTRO-moveit \
    # install catkin
    ros-$ROS_DISTRO-catkin \
    python3-catkin-tools \
    # utils
    git \
    locate \
    aptitude \
    vim htop \
    curl wget \
    spacenavd \
    python-is-python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

### Ros Workspace ###
# Set up the workspace
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash \
               && mkdir -p ~/ros_ws/src \
               && cd ~/ros_ws/src \
               && git clone https://github.com/cambel/ur3.git ros-ur -b noetic-devel"

# Updating ROSDEP and installing dependencies
RUN cd ~/ros_ws \
    && rosinstall ~/ros_ws/src /opt/ros/$ROS_DISTRO src/ros-ur/dependencies.rosinstall \
    && apt-get update \
    && rosdep fix-permissions \
    && rosdep update \
    && rosdep install --from-paths src --ignore-src --rosdistro=$ROS_DISTRO -y \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONIOENCODING UTF-8

## Python3 libraries ##
RUN python3 -m pip install pip --upgrade && \
    pip install matplotlib==2.2.3 spicy protobuf pyyaml pyquaternion rospkg \
    lxml tqdm catkin-pkg empy PyVirtualDisplay defusedxml gym psutil pyprind


# # Compiling ros workspace
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash \
               && cd ~/ros_ws/ \
               && rm -rf build \
               && catkin build"


RUN apt update -qq &&\
    cd /root/catkin_ws && \
    rosdep update && \
    rosdep install --rosdistro noetic --ignore-src --from-paths src -y 

RUN cd /root/catkin_ws && \
    /bin/bash -c 'source /opt/ros/noetic/setup.bash && catkin_make'

RUN echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc

# Download gazebo models only once
RUN mkdir -p ~/.gazebo
RUN git clone https://github.com/osrf/gazebo_models ~/.gazebo/models

# setup environment
EXPOSE 11345

RUN echo 'source /opt/ros/$ROS_DISTRO/setup.bash' >> ~/.bashrc
RUN echo 'source ~/ros_ws/devel/setup.bash' >> ~/.bashrc

RUN echo 'source /usr/share/gazebo/setup.sh' >> ~/.bashrc
RUN echo 'export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:~/ros_ws/src/ros-universal-robots/ur_gripper_gazebo/models/' >> ~/.bashrc

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

ENTRYPOINT []

CMD ["/bin/bash"]