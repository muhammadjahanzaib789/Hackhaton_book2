# Physical AI Book - ROS 2 + Gazebo Development Environment
# Base: ROS 2 Humble with Gazebo Sim

FROM osrf/ros:humble-desktop-full

LABEL maintainer="Physical AI Book Contributors"
LABEL description="Development environment for Physical AI & Humanoid Robotics"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    # Python tools
    python3-pip \
    python3-venv \
    # Development tools
    git \
    curl \
    wget \
    vim \
    # Gazebo dependencies
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    # Navigation
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    # MoveIt 2
    ros-humble-moveit \
    # Visualization
    ros-humble-rviz2 \
    ros-humble-rqt* \
    # Additional ROS 2 packages
    ros-humble-joint-state-publisher \
    ros-humble-robot-state-publisher \
    ros-humble-xacro \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for LLM integration
RUN pip3 install --no-cache-dir \
    ollama \
    openai \
    anthropic \
    pydantic \
    numpy \
    opencv-python \
    torch \
    transformers

# Create workspace
WORKDIR /workspace

# Set up ROS 2 environment
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
RUN echo "source /workspace/install/setup.bash 2>/dev/null || true" >> /root/.bashrc

# Expose ports for ROS 2 communication
EXPOSE 11311
EXPOSE 11345

# Default command
CMD ["/bin/bash"]
