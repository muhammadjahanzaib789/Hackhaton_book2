---
sidebar_position: 2
title: ROS 2 Installation
description: Setting up ROS 2 Humble on Ubuntu or Docker
---

# ROS 2 Installation

## Learning Objectives

By the end of this lesson, you will:

1. Have a working ROS 2 Humble installation
2. Understand the ROS 2 workspace structure
3. Be able to source and verify your environment
4. Run your first ROS 2 command

## Installation Options

Choose your installation method:

| Method | Best For | Time | Difficulty |
|--------|----------|------|------------|
| **Docker** | Any OS, quick start | 10 min | Easy |
| **Native Ubuntu** | Full development | 30 min | Medium |
| **WSL2** | Windows users | 45 min | Medium |

## Option 1: Docker (Recommended for Beginners)

Docker provides a consistent environment regardless of your host OS.

### Prerequisites

1. Install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/)
2. Ensure Docker is running

### Quick Start

```bash
# Pull our pre-configured image
docker pull osrf/ros:humble-desktop-full

# Run interactive container with GUI support (Linux)
docker run -it \
  --env="DISPLAY" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --name ros2_dev \
  osrf/ros:humble-desktop-full

# Run on Windows/macOS (no GUI)
docker run -it --name ros2_dev osrf/ros:humble-desktop-full
```

### Using Our Docker Compose

If you cloned the book repository:

```bash
cd physical-ai-book
docker compose up ros2 -d
docker compose exec ros2 bash
```

### Verify Installation

Inside the container:

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Check version
ros2 --version
# Expected output: ros2 0.x.x

# List available commands
ros2 --help
```

## Option 2: Native Ubuntu 22.04 Installation

For full development capabilities, install ROS 2 directly.

### Step 1: Set Locale

```bash
locale  # Check for UTF-8

sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

locale  # Verify settings
```

### Step 2: Add ROS 2 Repository

```bash
# Ensure Universe repository is enabled
sudo apt install software-properties-common
sudo add-apt-repository universe

# Add ROS 2 GPG key
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add repository to sources list
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
  sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### Step 3: Install ROS 2 Humble

```bash
sudo apt update
sudo apt upgrade

# Desktop install (recommended) - includes RViz, demos, tutorials
sudo apt install ros-humble-desktop

# Or full install - includes simulation
sudo apt install ros-humble-desktop-full
```

### Step 4: Install Development Tools

```bash
# Build tools
sudo apt install python3-colcon-common-extensions

# Additional useful packages
sudo apt install \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-joint-state-publisher \
  ros-humble-robot-state-publisher \
  ros-humble-xacro \
  python3-pip
```

### Step 5: Environment Setup

```bash
# Add to your .bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Verify
ros2 --version
```

## Option 3: WSL2 on Windows

Windows users can run ROS 2 through WSL2.

### Step 1: Install WSL2

```powershell
# Run in PowerShell as Administrator
wsl --install -d Ubuntu-22.04
```

### Step 2: Configure WSL2

Restart your computer, then open Ubuntu from Start menu.

```bash
# Update Ubuntu
sudo apt update && sudo apt upgrade -y
```

### Step 3: Install ROS 2

Follow the Native Ubuntu instructions above inside WSL2.

### Step 4: GUI Support (Optional)

For RViz and Gazebo visualization:

1. Install [VcXsrv](https://sourceforge.net/projects/vcxsrv/) on Windows
2. Start VcXsrv with "Disable access control" checked
3. In WSL2:

```bash
echo "export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0" >> ~/.bashrc
source ~/.bashrc
```

## Workspace Structure

ROS 2 uses **workspaces** to organize packages:

```
~/ros2_ws/              # Your workspace root
├── src/                # Source code goes here
│   ├── my_package/     # Your custom package
│   └── another_pkg/    # Another package
├── build/              # Build artifacts (auto-generated)
├── install/            # Installed packages (auto-generated)
└── log/                # Build logs (auto-generated)
```

### Create Your First Workspace

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build (even empty workspace)
colcon build

# Source the workspace
source install/setup.bash
```

### Workspace Sourcing Order

Always source in this order:

```bash
# 1. ROS 2 base installation
source /opt/ros/humble/setup.bash

# 2. Your workspace (overlays the base)
source ~/ros2_ws/install/setup.bash
```

:::tip Pro Tip
Add both to your `.bashrc` for automatic sourcing:
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "source ~/ros2_ws/install/setup.bash 2>/dev/null || true" >> ~/.bashrc
```
:::

## Verify Your Installation

Run these commands to ensure everything works:

### Test 1: ROS 2 Version

```bash
ros2 --version
```

**Expected output**: `ros2 0.x.x` (version number varies)

### Test 2: List Available Packages

```bash
ros2 pkg list | head -20
```

**Expected output**: List of installed ROS 2 packages

### Test 3: Run Demo Nodes

Open **two terminals**.

**Terminal 1** (Talker):
```bash
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_cpp talker
```

**Terminal 2** (Listener):
```bash
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_cpp listener
```

**Expected output**:
- Talker: `[INFO] [talker]: Publishing: 'Hello World: 1'`
- Listener: `[INFO] [listener]: I heard: [Hello World: 1]`

### Test 4: View Topics

In a **third terminal**:

```bash
source /opt/ros/humble/setup.bash
ros2 topic list
```

**Expected output**:
```
/chatter
/parameter_events
/rosout
```

## Troubleshooting

### "Command not found: ros2"

```bash
# Ensure ROS 2 is sourced
source /opt/ros/humble/setup.bash

# Check if installed
dpkg -l | grep ros-humble
```

### "Cannot locate package"

```bash
# Update package cache
sudo apt update

# Reinstall ros-humble-desktop
sudo apt install --reinstall ros-humble-desktop
```

### Docker: Display Issues

```bash
# Linux: Allow X server connections
xhost +local:docker

# Then restart container
docker restart ros2_dev
```

### WSL2: GUI Not Working

1. Ensure VcXsrv is running with "Disable access control"
2. Check DISPLAY variable: `echo $DISPLAY`
3. Test with: `xclock` (install with `sudo apt install x11-apps`)

## Summary

You now have:

- ✅ ROS 2 Humble installed
- ✅ Workspace structure created
- ✅ Environment properly sourced
- ✅ Verified with demo nodes

## Checkpoint

Before proceeding, verify:

1. `ros2 --version` returns a version number
2. `ros2 topic list` shows topics when talker/listener run
3. You understand the workspace directory structure

---

**Next**: [Lesson 3: Nodes and Topics](/docs/chapter-01-ros2-fundamentals/lesson-03-nodes-topics)
