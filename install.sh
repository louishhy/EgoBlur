#!/bin/bash

# EgoBlur Installation Script for UV
# This script helps users install EgoBlur with the appropriate CUDA support

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect CUDA version
detect_cuda_version() {
    if command_exists nvidia-smi; then
        local cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
        if [ -n "$cuda_version" ]; then
            echo "$cuda_version"
        else
            echo "unknown"
        fi
    else
        echo "none"
    fi
}

# Function to check if CUDA is available
check_cuda_availability() {
    if command_exists nvidia-smi; then
        nvidia-smi >/dev/null 2>&1
        return $?
    else
        return 1
    fi
}

# Function to install system dependencies
install_system_deps() {
    print_info "Installing system dependencies..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command_exists apt-get; then
            print_info "Installing dependencies via apt-get..."
            sudo apt-get update
            sudo apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0
        elif command_exists yum; then
            print_info "Installing dependencies via yum..."
            sudo yum install -y ffmpeg
        elif command_exists dnf; then
            print_info "Installing dependencies via dnf..."
            sudo dnf install -y ffmpeg
        else
            print_warning "Could not detect package manager. Please install ffmpeg manually."
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command_exists brew; then
            print_info "Installing dependencies via Homebrew..."
            brew install ffmpeg
        else
            print_warning "Homebrew not found. Please install ffmpeg manually."
        fi
    else
        print_warning "Unsupported OS. Please install ffmpeg manually."
    fi
}

# Function to install EgoBlur with UV
install_egoblur() {
    local cuda_version=$1
    local install_cpu=$2
    
    print_info "Installing EgoBlur with UV..."
    
    # Initialize UV project if not already done
    if [ ! -f "pyproject.toml" ]; then
        print_error "pyproject.toml not found. Please run this script from the EgoBlur root directory."
        exit 1
    fi
    
    # Install all dependencies
    print_info "Installing all dependencies..."
    uv sync
    
    print_success "EgoBlur installation completed!"
}

# Function to verify installation
verify_installation() {
    print_info "Verifying installation..."
    
    # Test PyTorch installation
    if uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"; then
        print_success "PyTorch installation verified!"
    else
        print_error "PyTorch installation verification failed!"
        return 1
    fi
    
    # Test other dependencies
    if uv run python -c "import cv2, numpy, moviepy; print('All dependencies imported successfully!')"; then
        print_success "All dependencies verified!"
    else
        print_error "Dependency verification failed!"
        return 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --cpu              Install CPU-only version (no CUDA)"
    echo "  --cuda-version X    Specify CUDA version (11.8, 12.1, 12.4)"
    echo "  --no-system-deps   Skip system dependency installation"
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Auto-detect CUDA and install"
    echo "  $0 --cpu                     # Install CPU-only version"
    echo "  $0 --cuda-version 12.1       # Install with specific CUDA version"
}

# Main function
main() {
    local install_cpu=false
    local cuda_version=""
    local skip_system_deps=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --cpu)
                install_cpu=true
                shift
                ;;
            --cuda-version)
                cuda_version="$2"
                shift 2
                ;;
            --no-system-deps)
                skip_system_deps=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    print_info "EgoBlur Installation Script"
    print_info "============================"
    
    # Check if UV is installed
    if ! command_exists uv; then
        print_error "UV is not installed. Please install UV first:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    
    # Detect CUDA version if not specified
    if [ -z "$cuda_version" ] && [ "$install_cpu" = false ]; then
        if check_cuda_availability; then
            cuda_version=$(detect_cuda_version)
            if [ "$cuda_version" != "none" ] && [ "$cuda_version" != "unknown" ]; then
                print_info "Detected CUDA version: $cuda_version"
            else
                print_warning "Could not detect CUDA version. Installing CPU-only version."
                install_cpu=true
            fi
        else
            print_info "CUDA not available. Installing CPU-only version."
            install_cpu=true
        fi
    fi
    
    # Install system dependencies
    if [ "$skip_system_deps" = false ]; then
        install_system_deps
    fi
    
    # Install EgoBlur
    install_egoblur "$cuda_version" "$install_cpu"
    
    # Verify installation
    if verify_installation; then
        print_success "Installation completed successfully!"
        echo ""
        print_info "You can now run EgoBlur demo with:"
        echo "  uv run python script/demo_ego_blur.py --help"
        echo ""
        print_info "Or use the CLI command:"
        echo "  uv run egoblur-demo --help"
    else
        print_error "Installation verification failed!"
        exit 1
    fi
}

# Run main function
main "$@"
