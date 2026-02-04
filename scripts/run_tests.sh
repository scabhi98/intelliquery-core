#!/bin/bash
# Run tests for LexiQuery services (Linux/Mac)

set -e  # Exit on error

echo "=== LexiQuery Test Runner ==="
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
SERVICE=""
COVERAGE=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--service)
            SERVICE="$2"
            shift 2
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [-s SERVICE] [-c] [-v]"
            echo "  -s, --service    Run tests for specific service"
            echo "  -c, --coverage   Generate coverage report"
            echo "  -v, --verbose    Verbose output"
            exit 1
            ;;
    esac
done

# Function to run tests for a service
run_service_tests() {
    local service_name=$1
    local service_dir="$ROOT_DIR/services/$service_name"
    
    echo -e "${YELLOW}Testing: $service_name${NC}"
    
    if [ ! -d "$service_dir" ]; then
        echo -e "${RED}  Error: Service directory not found: $service_dir${NC}"
        return 1
    fi
    
    if [ ! -d "$service_dir/.venv" ]; then
        echo -e "${RED}  Error: Virtual environment not found. Run setup_envs.sh first${NC}"
        return 1
    fi
    
    cd "$service_dir"
    source .venv/bin/activate
    
    # Build pytest command
    local pytest_cmd="pytest tests/"
    
    if [ "$VERBOSE" = true ]; then
        pytest_cmd="$pytest_cmd -v"
    fi
    
    if [ "$COVERAGE" = true ]; then
        pytest_cmd="$pytest_cmd --cov=src --cov-report=html --cov-report=term"
    fi
    
    echo "  Running: $pytest_cmd"
    
    if $pytest_cmd; then
        echo -e "${GREEN}  ✓ Tests passed${NC}"
        
        if [ "$COVERAGE" = true ]; then
            echo -e "${GREEN}  Coverage report: $service_dir/htmlcov/index.html${NC}"
        fi
    else
        echo -e "${RED}  ✗ Tests failed${NC}"
        deactivate
        return 1
    fi
    
    deactivate
    echo ""
}

# Run tests for specific service or all services
if [ -n "$SERVICE" ]; then
    run_service_tests "$SERVICE"
else
    services=(
        "core-engine"
        "protocol-interface"
        "sop-engine"
        "query-generator"
        "cache-learning"
        "data-connectors"
        "cost-tracker"
    )
    
    FAILED_SERVICES=()
    
    for service in "${services[@]}"; do
        if ! run_service_tests "$service"; then
            FAILED_SERVICES+=("$service")
        fi
    done
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    
    if [ ${#FAILED_SERVICES[@]} -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
    else
        echo -e "${RED}Some tests failed:${NC}"
        for service in "${FAILED_SERVICES[@]}"; do
            echo -e "${RED}  - $service${NC}"
        done
        echo -e "${GREEN}========================================${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}========================================${NC}"
fi

echo ""
