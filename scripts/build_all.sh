#!/bin/bash
# Build Docker images for all LexiQuery services (Linux/Mac)

set -e  # Exit on error

echo "=== LexiQuery Docker Build ==="
echo ""

# Get the root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
TAG="latest"
REGISTRY=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    exit 1
fi

echo -e "${GREEN}Using Docker: $(docker --version)${NC}"
echo ""

# Function to build a service image
build_service() {
    local service_name=$1
    local service_dir="$ROOT_DIR/services/$service_name"
    local image_name="lexiquery-$service_name"
    
    if [ -n "$REGISTRY" ]; then
        image_name="$REGISTRY/$image_name"
    fi
    
    echo -e "${YELLOW}Building: $service_name${NC}"
    
    if [ ! -f "$service_dir/Dockerfile" ]; then
        echo -e "${RED}  Error: Dockerfile not found in $service_dir${NC}"
        return 1
    fi
    
    cd "$ROOT_DIR"
    
    docker build \
        -t "$image_name:$TAG" \
        -t "$image_name:latest" \
        -f "$service_dir/Dockerfile" \
        . \
        --build-arg SERVICE_NAME="$service_name"
    
    echo -e "${GREEN}  ✓ Built: $image_name:$TAG${NC}"
    echo ""
}

# Services to build
services=(
    "core-engine"
    "protocol-interface"
    "sop-engine"
    "query-generator"
    "cache-learning"
    "data-connectors"
    "cost-tracker"
)

# Build all services
for service in "${services[@]}"; do
    build_service "$service"
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All images built successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Built images:"
for service in "${services[@]}"; do
    if [ -n "$REGISTRY" ]; then
        echo "  - $REGISTRY/lexiquery-$service:$TAG"
    else
        echo "  - lexiquery-$service:$TAG"
    fi
done
echo ""
echo "To push images to registry:"
echo "  docker push <image-name>:<tag>"
echo ""
