#!/bin/bash
# ==============================================================================
# VRAG/VimRAG Demo Launch Script
# ==============================================================================
# This script helps you launch the VRAG or VimRAG demo applications.
#
# VRAG Demo: Uses a local vLLM server with the VRAG model + local search engine
# VimRAG Demo: Uses Qwen API (DashScope) + local search engine, features DAG
#              visualization, multimodal memory graph, and thinking mode
#
# Prerequisites:
#   - Python environment with dependencies installed (pip install -r requirements.txt)
#   - For VRAG: GPU with sufficient VRAM (A100 80G recommended for 7B model)
#   - For VimRAG: Valid DASHSCOPE_API_KEY environment variable
#
# Usage:
#   ./run_demo.sh vrag     # Launch VRAG demo (local model)
#   ./run_demo.sh vimrag   # Launch VimRAG demo (Qwen API)
#   ./run_demo.sh search   # Launch search engine only
#   ./run_demo.sh help     # Show this help message
# ==============================================================================

set -e

# ==============================================================================
# Configuration Variables
# ==============================================================================

# Search Engine Configuration
SEARCH_ENGINE_HOST="0.0.0.0"
SEARCH_ENGINE_PORT=8001

# vLLM Server Configuration (for VRAG demo)
VLLM_HOST="0.0.0.0"
VLLM_PORT=8002
VLLM_MODEL="autumncc/Qwen2.5-VL-7B-VRAG"
VLLM_SERVED_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
VLLM_MAX_IMAGES=10

# Streamlit Configuration
STREAMLIT_PORT=8501

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

print_info() {
    echo "[INFO] $1"
}

print_warning() {
    echo "[WARNING] $1"
}

print_error() {
    echo "[ERROR] $1" >&2
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "Command '$1' not found. Please install it first."
        exit 1
    fi
}

wait_for_service() {
    local host=$1
    local port=$2
    local name=$3
    local max_attempts=60
    local attempt=1
    
    print_info "Waiting for $name to be ready at $host:$port..."
    while ! nc -z "$host" "$port" 2>/dev/null; do
        if [ $attempt -ge $max_attempts ]; then
            print_error "$name failed to start within ${max_attempts} seconds"
            return 1
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    print_info "$name is ready!"
}

show_help() {
    echo "VRAG/VimRAG Demo Launch Script"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  vrag      Launch VRAG demo (local vLLM model + search engine + Streamlit)"
    echo "  vimrag    Launch VimRAG demo (Qwen API + search engine + Streamlit)"
    echo "  search    Launch search engine only"
    echo "  help      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  DASHSCOPE_API_KEY    Required for VimRAG demo (Qwen API authentication)"
    echo "  CUDA_VISIBLE_DEVICES Set GPU device for vLLM (default: auto)"
    echo ""
    echo "Examples:"
    echo "  # Launch VRAG demo on GPU 0 for search, GPU 1 for vLLM"
    echo "  CUDA_VISIBLE_DEVICES=0 ./run_demo.sh search &"
    echo "  CUDA_VISIBLE_DEVICES=1 vllm serve $VLLM_MODEL --port $VLLM_PORT --host $VLLM_HOST"
    echo "  streamlit run demo/app.py"
    echo ""
    echo "  # Launch VimRAG demo"
    echo "  export DASHSCOPE_API_KEY=your_api_key"
    echo "  ./run_demo.sh vimrag"
    echo ""
    echo "Port Configuration:"
    echo "  Search Engine: $SEARCH_ENGINE_PORT"
    echo "  vLLM Server:   $VLLM_PORT"
    echo "  Streamlit:     $STREAMLIT_PORT"
}

# ==============================================================================
# Service Launch Functions
# ==============================================================================

launch_search_engine() {
    print_header "Launching Search Engine"
    print_info "Starting search engine on port $SEARCH_ENGINE_PORT..."
    print_info "Using embedding model for visual retrieval"
    
    # Launch search engine in background
    python search_engine/search_engine_api.py &
    SEARCH_PID=$!
    
    # Wait for search engine to be ready
    sleep 3
    if ! kill -0 $SEARCH_PID 2>/dev/null; then
        print_error "Search engine failed to start"
        exit 1
    fi
    
    print_info "Search engine started (PID: $SEARCH_PID)"
    echo $SEARCH_PID
}

launch_vllm_server() {
    print_header "Launching vLLM Server"
    print_info "Model: $VLLM_MODEL"
    print_info "Served as: $VLLM_SERVED_NAME"
    print_info "Port: $VLLM_PORT"
    print_info "Max images per prompt: $VLLM_MAX_IMAGES"
    
    # Check if vllm is installed
    check_command "vllm"
    
    # Launch vLLM server
    vllm serve "$VLLM_MODEL" \
        --port "$VLLM_PORT" \
        --host "$VLLM_HOST" \
        --limit-mm-per-prompt "image=$VLLM_MAX_IMAGES" \
        --served-model-name "$VLLM_SERVED_NAME" &
    
    VLLM_PID=$!
    print_info "vLLM server starting (PID: $VLLM_PID)"
    print_info "This may take a few minutes to load the model..."
    echo $VLLM_PID
}

launch_vrag_demo() {
    print_header "Launching VRAG Streamlit Demo"
    print_info "Starting Streamlit on port $STREAMLIT_PORT..."
    
    streamlit run demo/app.py --server.port "$STREAMLIT_PORT"
}

launch_vimrag_demo() {
    print_header "Launching VimRAG Streamlit Demo"
    
    # Check for API key
    if [ -z "$DASHSCOPE_API_KEY" ]; then
        print_warning "DASHSCOPE_API_KEY is not set!"
        print_info "You can set it in the Streamlit sidebar, or export it:"
        print_info "  export DASHSCOPE_API_KEY=your_api_key"
    else
        print_info "DASHSCOPE_API_KEY is configured"
    fi
    
    print_info "Starting Streamlit on port $STREAMLIT_PORT..."
    print_info "VimRAG uses Qwen3.5-VL-Plus via DashScope API"
    
    streamlit run demo/vimrag_app.py --server.port "$STREAMLIT_PORT"
}

# ==============================================================================
# Main Entry Point
# ==============================================================================

main() {
    case "${1:-help}" in
        vrag)
            print_header "VRAG Demo - Full Stack Launch"
            print_info "This will launch: Search Engine + vLLM Server + Streamlit Demo"
            print_warning "Make sure you have sufficient GPU memory!"
            echo ""
            
            # Check dependencies
            check_command "python"
            check_command "streamlit"
            
            # Launch services
            SEARCH_PID=$(launch_search_engine)
            sleep 2
            
            VLLM_PID=$(launch_vllm_server)
            
            # Wait for vLLM to be ready (it takes time to load the model)
            print_info "Waiting for vLLM server to load the model (this may take a few minutes)..."
            sleep 30
            
            # Launch Streamlit demo
            launch_vrag_demo
            
            # Cleanup on exit
            trap "kill $SEARCH_PID $VLLM_PID 2>/dev/null" EXIT
            ;;
            
        vimrag)
            print_header "VimRAG Demo - API-based Launch"
            print_info "This will launch: Search Engine + Streamlit Demo"
            print_info "VimRAG connects to Qwen API (DashScope) for model inference"
            echo ""
            
            # Check dependencies
            check_command "python"
            check_command "streamlit"
            
            # Launch search engine first
            SEARCH_PID=$(launch_search_engine)
            sleep 3
            
            # Launch VimRAG Streamlit demo
            launch_vimrag_demo
            
            # Cleanup on exit
            trap "kill $SEARCH_PID 2>/dev/null" EXIT
            ;;
            
        search)
            print_header "Search Engine Only"
            print_info "Launching search engine service..."
            
            check_command "python"
            
            # Run in foreground
            python search_engine/search_engine_api.py
            ;;
            
        help|--help|-h)
            show_help
            ;;
            
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
