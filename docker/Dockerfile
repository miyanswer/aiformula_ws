FROM ros:humble

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install ROS 2 desktop packages and essential development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    bash-completion \
    build-essential \
    cmake \
    curl \
    gdb \
    git \
    htop \
    iputils-ping \
    nano \
    net-tools \
    novnc \
    python3-argcomplete \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    python3-vcstool \
    sudo \
    tmux \
    tree \
    vim \
    websockify \
    wget \
    fluxbox \
    xvfb \
    x11vnc \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Setup noVNC default index
RUN ln -s /usr/share/novnc/vnc.html /usr/share/novnc/index.html 2>/dev/null || true


# Initialize and update rosdep
RUN if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then \
        rosdep init; \
    fi

# Create non-root user 'rosuser' with sudo privileges
ARG USERNAME=rosuser
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME 2>/dev/null || true \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash 2>/dev/null || true \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Update rosdep as rosuser
USER $USERNAME
RUN rosdep update
USER root

# Configure system-wide ROS 2 environment for all shells (interactive, non-interactive, and login)
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc \
    && echo "if [ -f /aiformula_ws/install/setup.bash ]; then source /aiformula_ws/install/setup.bash; fi" >> /etc/bash.bashrc \
    && mkdir -p /etc/profile.d \
    && echo "source /opt/ros/humble/setup.bash" > /etc/profile.d/ros.sh \
    && echo "if [ -f /aiformula_ws/install/setup.bash ]; then source /aiformula_ws/install/setup.bash; fi" >> /etc/profile.d/ros.sh \
    && chmod +x /etc/profile.d/ros.sh

# Configure .bashrc for rosuser (prepend to ensure it runs even for non-interactive subshells)
RUN sed -i '1i source /opt/ros/humble/setup.bash' /home/$USERNAME/.bashrc \
    && echo "if [ -f /aiformula_ws/install/setup.bash ]; then source /aiformula_ws/install/setup.bash; fi" >> /home/$USERNAME/.bashrc \
    && echo "eval \"\$(register-python-argcomplete3 ros2)\"" >> /home/$USERNAME/.bashrc \
    && echo "eval \"\$(register-python-argcomplete3 colcon)\"" >> /home/$USERNAME/.bashrc \
    && echo "export ROS_DOMAIN_ID=0" >> /home/$USERNAME/.bashrc

# Set default shell environment variables
ENV ROS_DISTRO=humble
ENV ROS_ROOT=/opt/ros/humble
ENV ROS_VERSION=2
ENV ROS_PYTHON_VERSION=3
ENV PATH=/opt/ros/humble/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/ros/humble/lib:$LD_LIBRARY_PATH
ENV AMENT_PREFIX_PATH=/opt/ros/humble
ENV PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:$PYTHONPATH
ENV CMAKE_PREFIX_PATH=/opt/ros/humble:$CMAKE_PREFIX_PATH
ENV DISPLAY=:1
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV QT_X11_NO_MITSHM=1


# Set up entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Workspace directory
WORKDIR /aiformula_ws
RUN chown -R $USERNAME:$USERNAME /aiformula_ws 2>/dev/null || true

USER $USERNAME

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
