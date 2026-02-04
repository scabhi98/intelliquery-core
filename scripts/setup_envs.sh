#!/bin/bash
# Setup virtual environments for all LexiQuery services (Linux/Mac)

set -e  # Exit on error

echo "=== LexiQuery Environment Setup ==="
echo ""

# Get the root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_CMD=""
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 11 ]; then
        PYTHON_CMD="python3"
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}Error: Python 3.11 or higher is required${NC}"
    echo "Please install Python 3.11+ and try again"
    exit 1
fi

echo -e "${GREEN}Using: $PYTHON_CMD ($($PYTHON_CMD --version))${NC}"
echo ""

# Function to setup a service environment
setup_service() {
    local service_name=$1
    local service_dir="$ROOT_DIR/services/$service_name"
    
    echo -e "${YELLOW}Setting up: $service_name${NC}"
    
    if [ ! -d "$service_dir" ]; then
        echo -e "${RED}Error: Service directory not found: $service_dir${NC}"
        return 1
    fi
    
    cd "$service_dir"
    
    # Remove existing venv if present
    if [ -d ".venv" ]; then
        echo "  Removing existing .venv..."
        rm -rf .venv
    fi
    
    # Create virtual environment
    echo "  Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    
    # Activate and install dependencies
    echo "  Installing dependencies..."
    source .venv/bin/activate
    
    pip install --upgrade pip --quiet
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt --quiet
    else
        echo -e "${YELLOW}  Warning: No requirements.txt found${NC}"
    fi
    
    deactivate
    
    echo -e "${GREEN}  ✓ Complete${NC}"
    echo ""
}

# Setup shared environment first
echo -e "${YELLOW}Setting up: shared (common dependencies)${NC}"
cd "$ROOT_DIR/shared"

if [ -d ".venv" ]; then
    echo "  Removing existing .venv..."
    rm -rf .venv
fi

echo "  Creating virtual environment..."
$PYTHON_CMD -m venv .venv

echo "  Installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip --quiet

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
else
    echo -e "${YELLOW}  Warning: No requirements.txt found${NC}"
fi

deactivate
echo -e "${GREEN}  ✓ Complete${NC}"
echo ""

# Setup all services
services=(
    "core-engine"
    "protocol-interface"
    "sop-engine"
    "query-generator"
    "cache-learning"
    "data-connectors"
    "cost-tracker"
)

for service in "${services[@]}"; do
    setup_service "$service"
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All environments setup successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To activate a service environment:"
echo "  cd services/<service-name>"
echo "  source .venv/bin/activate"
echo ""
echo "Available services:"
for service in "${services[@]}"; do
    echo "  - $service"
done
echo ""
