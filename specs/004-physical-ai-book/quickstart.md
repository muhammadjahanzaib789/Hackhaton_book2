# Quickstart Guide: Physical AI & Humanoid Robotics Book

**Branch**: `004-physical-ai-book` | **Date**: 2025-12-15

## Prerequisites

Before starting, ensure you have:

- **Operating System**: Ubuntu 22.04 LTS (or Docker on Windows/macOS)
- **Python**: 3.10 or higher
- **Node.js**: 18.x or higher (for Docusaurus)
- **Git**: Latest version
- **Hardware**: 8GB RAM minimum, 16GB recommended

## Quick Setup (5 Minutes)

### Option A: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/[org]/physical-ai-book.git
cd physical-ai-book

# Start development environment
docker compose up -d

# Open in browser
open http://localhost:3000
```

### Option B: Local Installation

```bash
# Clone the repository
git clone https://github.com/[org]/physical-ai-book.git
cd physical-ai-book

# Install Node.js dependencies
npm install

# Start development server
npm start
```

## ROS 2 Development Environment

For code examples that require ROS 2:

### Using Docker (Easiest)

```bash
# Pull ROS 2 + Gazebo development image
docker pull osrf/ros:humble-desktop-full

# Run with GUI support (Linux)
docker run -it \
  --env="DISPLAY" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="$(pwd):/workspace" \
  osrf/ros:humble-desktop-full

# Inside container
cd /workspace
source /opt/ros/humble/setup.bash
```

### Native Installation (Ubuntu 22.04)

```bash
# Add ROS 2 repository
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Humble
sudo apt update
sudo apt install -y ros-humble-desktop

# Install Gazebo
sudo apt install -y ros-humble-gazebo-ros-pkgs

# Source ROS 2
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## LLM Setup (Ollama - Local)

For chapters involving LLM integration:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull recommended model
ollama pull llama3.2

# Verify installation
ollama run llama3.2 "Hello, I am a humanoid robot."
```

## Verify Installation

Run the verification script:

```bash
# From repository root
./scripts/verify-setup.sh
```

Expected output:
```
✓ Node.js 18.x detected
✓ npm packages installed
✓ Docusaurus builds successfully
✓ ROS 2 Humble detected
✓ Gazebo Sim available
✓ Ollama running with llama3.2
✓ All systems ready!
```

## Project Structure

```
physical-ai-book/
├── docs/                    # Markdown lesson content
│   ├── intro.md
│   ├── chapter-01-ros2-fundamentals/
│   ├── chapter-02-simulation/
│   ├── chapter-03-perception/
│   ├── chapter-04-navigation/
│   ├── chapter-05-manipulation/
│   ├── chapter-06-llm-integration/
│   ├── chapter-07-vla-pipelines/
│   └── chapter-08-capstone/
├── static/
│   ├── img/                 # Images and diagrams
│   └── models/              # URDF/SDF robot models
├── src/
│   ├── components/          # React components for book
│   └── examples/            # Runnable code examples
│       ├── ros2/            # ROS 2 Python/C++ examples
│       ├── llm/             # LLM integration examples
│       └── capstone/        # Capstone project code
├── docusaurus.config.js     # Site configuration
├── sidebars.js              # Navigation structure
├── package.json             # Node.js dependencies
└── docker-compose.yml       # Development environment
```

## Running Examples

### Basic ROS 2 Example (Chapter 1)

```bash
# Terminal 1: Start a simple publisher
cd src/examples/ros2/ch01
python3 simple_publisher.py

# Terminal 2: Start a subscriber
python3 simple_subscriber.py

# Expected: Subscriber receives messages from publisher
```

### Gazebo Simulation (Chapter 2)

```bash
# Launch humanoid robot in Gazebo
ros2 launch physical_ai_bringup humanoid_gazebo.launch.py

# In another terminal, send a movement command
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5}}"
```

### LLM Action Planning (Chapter 6)

```bash
# Start the LLM action planner
cd src/examples/llm
python3 action_planner.py

# In another terminal, send a command
python3 -c "
from action_planner import RobotActionPlanner
planner = RobotActionPlanner()
result = planner.plan_action('bring me the red cup')
print(result)
"
```

## Common Issues

### Docker: Cannot connect to display

```bash
# Linux: Allow Docker to access X server
xhost +local:docker
```

### ROS 2: Package not found

```bash
# Rebuild workspace
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### Ollama: Model not responding

```bash
# Restart Ollama service
ollama serve

# Check if model is loaded
ollama list
```

### Gazebo: Black screen

```bash
# Check GPU drivers
glxinfo | grep "OpenGL renderer"

# Use software rendering if needed
export LIBGL_ALWAYS_SOFTWARE=1
```

## Next Steps

1. **Start with Chapter 1**: ROS 2 Fundamentals
2. **Complete exercises**: Each chapter has hands-on exercises
3. **Build incrementally**: Each chapter builds on the previous
4. **Join discussions**: GitHub Discussions for questions

## Support

- **Issues**: [GitHub Issues](https://github.com/[org]/physical-ai-book/issues)
- **Discussions**: [GitHub Discussions](https://github.com/[org]/physical-ai-book/discussions)
- **Updates**: Star the repository for notifications

---

**Estimated setup time**: 15-30 minutes (Docker) | 45-60 minutes (native)
