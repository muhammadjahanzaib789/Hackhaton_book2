#!/bin/bash
# Physical AI Book - Setup Verification Script
# Checks all required dependencies are installed

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Physical AI Book - Setup Verification"
echo "=========================================="
echo ""

ERRORS=0

# Check Node.js
check_node() {
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
        if [ "$NODE_VERSION" -ge 18 ]; then
            echo -e "${GREEN}✓${NC} Node.js $(node --version) detected"
        else
            echo -e "${RED}✗${NC} Node.js 18+ required, found $(node --version)"
            ERRORS=$((ERRORS+1))
        fi
    else
        echo -e "${RED}✗${NC} Node.js not found"
        ERRORS=$((ERRORS+1))
    fi
}

# Check npm
check_npm() {
    if command -v npm &> /dev/null; then
        echo -e "${GREEN}✓${NC} npm $(npm --version) detected"
    else
        echo -e "${RED}✗${NC} npm not found"
        ERRORS=$((ERRORS+1))
    fi
}

# Check if npm packages are installed
check_npm_packages() {
    if [ -d "node_modules" ]; then
        echo -e "${GREEN}✓${NC} npm packages installed"
    else
        echo -e "${YELLOW}!${NC} npm packages not installed (run 'npm install')"
    fi
}

# Check Docusaurus build
check_docusaurus() {
    if npm run build &> /dev/null; then
        echo -e "${GREEN}✓${NC} Docusaurus builds successfully"
    else
        echo -e "${RED}✗${NC} Docusaurus build failed"
        ERRORS=$((ERRORS+1))
    fi
}

# Check ROS 2
check_ros2() {
    if command -v ros2 &> /dev/null; then
        ROS_DISTRO=$(ros2 --version 2>&1 | head -1)
        echo -e "${GREEN}✓${NC} ROS 2 detected: $ROS_DISTRO"
    else
        echo -e "${YELLOW}!${NC} ROS 2 not found (optional for docs-only development)"
    fi
}

# Check Gazebo
check_gazebo() {
    if command -v gz &> /dev/null; then
        GZ_VERSION=$(gz --version 2>&1 | head -1)
        echo -e "${GREEN}✓${NC} Gazebo Sim detected: $GZ_VERSION"
    elif command -v gazebo &> /dev/null; then
        echo -e "${GREEN}✓${NC} Gazebo Classic detected"
    else
        echo -e "${YELLOW}!${NC} Gazebo not found (optional for docs-only development)"
    fi
}

# Check Ollama
check_ollama() {
    if command -v ollama &> /dev/null; then
        echo -e "${GREEN}✓${NC} Ollama detected"
        # Check if model is available
        if ollama list 2>/dev/null | grep -q "llama"; then
            echo -e "${GREEN}✓${NC} Ollama model available"
        else
            echo -e "${YELLOW}!${NC} No Ollama model found (run 'ollama pull llama3.2')"
        fi
    else
        echo -e "${YELLOW}!${NC} Ollama not found (optional for LLM chapters)"
    fi
}

# Check Python
check_python() {
    if command -v python3 &> /dev/null; then
        PY_VERSION=$(python3 --version | cut -d' ' -f2)
        PY_MAJOR=$(echo $PY_VERSION | cut -d'.' -f1)
        PY_MINOR=$(echo $PY_VERSION | cut -d'.' -f2)
        if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 10 ]; then
            echo -e "${GREEN}✓${NC} Python $PY_VERSION detected"
        else
            echo -e "${YELLOW}!${NC} Python 3.10+ recommended, found $PY_VERSION"
        fi
    else
        echo -e "${YELLOW}!${NC} Python 3 not found"
    fi
}

# Check Docker
check_docker() {
    if command -v docker &> /dev/null; then
        echo -e "${GREEN}✓${NC} Docker $(docker --version | cut -d' ' -f3 | tr -d ',') detected"
    else
        echo -e "${YELLOW}!${NC} Docker not found (optional)"
    fi
}

# Run all checks
echo "Checking core dependencies..."
echo ""
check_node
check_npm
check_npm_packages
echo ""

echo "Checking optional dependencies..."
echo ""
check_python
check_ros2
check_gazebo
check_ollama
check_docker
echo ""

# Summary
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All required dependencies satisfied!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run 'npm install' if packages not installed"
    echo "  2. Run 'npm start' to start development server"
    echo "  3. Open http://localhost:3000 in your browser"
else
    echo -e "${RED}$ERRORS required dependency issue(s) found${NC}"
    echo "Please resolve the issues above before proceeding."
    exit 1
fi
echo "=========================================="
