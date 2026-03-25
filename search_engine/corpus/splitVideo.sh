#!/usr/bin/env bash
#
# splitVideo.sh - Split long videos into fixed-duration chunks for indexing
#
# This script processes video files in a directory, converts non-mp4 formats
# to mp4, and splits long videos into smaller chunks suitable for video RAG
# indexing pipelines.
#

set -euo pipefail

# ============================================================================
# Color and logging utilities
# ============================================================================
readonly RED='\033[0;31m'
readonly YELLOW='\033[0;33m'
readonly GREEN='\033[0;32m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_progress() {
    echo -ne "${BLUE}[PROGRESS]${NC} $*\r"
}

# ============================================================================
# Default configuration
# ============================================================================
INPUT_DIR=""
OUTPUT_DIR="./search_engine/corpus/video_chunks"
CHUNK_DURATION=60
MIN_SOURCE_DURATION=1    # Minimum source video duration in seconds
MIN_CHUNK_DURATION=2     # Minimum chunk duration in seconds

# ============================================================================
# Usage information
# ============================================================================
usage() {
    cat <<EOF
Usage: $(basename "$0") -i INPUT_DIR [-o OUTPUT_DIR] [-d CHUNK_DURATION] [-h]

Split long videos into fixed-duration chunks for video RAG indexing.

Options:
    -i INPUT_DIR       Input directory containing video files (required)
    -o OUTPUT_DIR      Output directory for video chunks
                       (default: ./search_engine/corpus/video)
    -d CHUNK_DURATION  Duration of each chunk in seconds (default: 60)
    -h                 Show this help message and exit

Examples:
    # Basic usage with required input directory
    $(basename "$0") -i /path/to/videos

    # Specify custom output directory and chunk duration
    $(basename "$0") -i /path/to/videos -o /path/to/output -d 30

    # Process videos with 2-minute chunks
    $(basename "$0") -i ./raw_videos -d 120

Output:
    Video chunks are saved as {original_name}_chunk_{NNN}.mp4 where NNN
    is a zero-padded index starting from 001.

Notes:
    - Non-mp4 files are automatically converted to mp4 before processing
    - Source videos shorter than ${MIN_SOURCE_DURATION}s are skipped (not deleted)
    - Generated chunks shorter than ${MIN_CHUNK_DURATION}s are discarded
    - Source files are NEVER deleted

EOF
    exit 0
}

# ============================================================================
# Dependency checking
# ============================================================================
check_dependencies() {
    local missing=()
    
    if ! command -v ffmpeg &>/dev/null; then
        missing+=("ffmpeg")
    fi
    
    if ! command -v ffprobe &>/dev/null; then
        missing+=("ffprobe")
    fi
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required dependencies: ${missing[*]}"
        log_error "Please install them before running this script."
        log_error "  Ubuntu/Debian: sudo apt-get install ffmpeg"
        log_error "  macOS: brew install ffmpeg"
        log_error "  CentOS/RHEL: sudo yum install ffmpeg"
        exit 1
    fi
}

# ============================================================================
# Helper functions
# ============================================================================

# Get video duration in seconds using ffprobe (more reliable than parsing ffmpeg output)
# Uses awk for floating point arithmetic instead of bc
get_video_duration() {
    local file="$1"
    ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file" 2>/dev/null || echo "0"
}

# Compare two floating point numbers: returns 0 if $1 < $2
float_lt() {
    awk -v a="$1" -v b="$2" 'BEGIN { exit !(a < b) }'
}

# Add two floating point numbers
float_add() {
    awk -v a="$1" -v b="$2" 'BEGIN { printf "%.3f", a + b }'
}

# ============================================================================
# Parse command line arguments
# ============================================================================
parse_args() {
    while getopts ":i:o:d:h" opt; do
        case ${opt} in
            i)
                INPUT_DIR="$OPTARG"
                ;;
            o)
                OUTPUT_DIR="$OPTARG"
                ;;
            d)
                CHUNK_DURATION="$OPTARG"
                ;;
            h)
                usage
                ;;
            \?)
                log_error "Invalid option: -$OPTARG"
                echo "Use -h for help."
                exit 1
                ;;
            :)
                log_error "Option -$OPTARG requires an argument."
                echo "Use -h for help."
                exit 1
                ;;
        esac
    done
    
    # Validate required arguments
    if [[ -z "$INPUT_DIR" ]]; then
        log_error "Input directory is required. Use -i to specify."
        echo "Use -h for help."
        exit 1
    fi
    
    # Validate input directory exists
    if [[ ! -d "$INPUT_DIR" ]]; then
        log_error "Input directory does not exist: $INPUT_DIR"
        exit 1
    fi
    
    # Validate chunk duration is a positive number
    if ! [[ "$CHUNK_DURATION" =~ ^[0-9]+$ ]] || [[ "$CHUNK_DURATION" -le 0 ]]; then
        log_error "Chunk duration must be a positive integer: $CHUNK_DURATION"
        exit 1
    fi
}

# ============================================================================
# Process a single video file
# ============================================================================
process_video() {
    local file="$1"
    local filename
    local extension
    local base_name
    local total_seconds
    local temp_mp4=""
    local process_file
    local chunk_index=1
    local start_time=0
    
    filename=$(basename -- "$file")
    extension="${filename##*.}"
    extension=$(echo "$extension" | tr '[:upper:]' '[:lower:]')
    base_name="${filename%.*}"
    
    # Get video duration
    total_seconds=$(get_video_duration "$file")
    
    if [[ -z "$total_seconds" ]] || [[ "$total_seconds" == "0" ]]; then
        log_warn "Cannot get duration for: $filename (skipping)"
        return 1
    fi
    
    # Skip videos shorter than minimum duration (DO NOT DELETE)
    if float_lt "$total_seconds" "$MIN_SOURCE_DURATION"; then
        log_warn "Video shorter than ${MIN_SOURCE_DURATION}s: $filename (skipping)"
        return 1
    fi
    
    # Convert non-mp4 files to mp4 (create temp file)
    if [[ "$extension" != "mp4" ]]; then
        log_info "Converting to mp4: $filename"
        temp_mp4="${OUTPUT_DIR}/.temp_${base_name}_$$.mp4"
        if ! ffmpeg -i "$file" -c:v libx264 -c:a aac -strict experimental "$temp_mp4" -y -loglevel error 2>/dev/null; then
            log_warn "Failed to convert: $filename (skipping)"
            [[ -f "$temp_mp4" ]] && rm -f "$temp_mp4"
            return 1
        fi
        process_file="$temp_mp4"
    else
        process_file="$file"
    fi
    
    # Split video into chunks
    while float_lt "$start_time" "$total_seconds"; do
        # Generate output filename with zero-padded index
        local output_file
        output_file=$(printf "%s/%s_chunk_%03d.mp4" "$OUTPUT_DIR" "$base_name" "$chunk_index")
        
        # Split video chunk
        if ! ffmpeg -i "$process_file" -ss "$start_time" -t "$CHUNK_DURATION" -c copy "$output_file" -y -loglevel error 2>/dev/null; then
            log_warn "Failed to create chunk: $output_file"
            [[ -f "$output_file" ]] && rm -f "$output_file"
        else
            # Validate chunk duration
            local chunk_seconds
            chunk_seconds=$(get_video_duration "$output_file")
            
            if [[ -z "$chunk_seconds" ]] || float_lt "$chunk_seconds" "$MIN_CHUNK_DURATION"; then
                # Remove generated chunk if too short (this is safe - it's our generated file)
                rm -f "$output_file"
            fi
        fi
        
        # Update for next iteration
        start_time=$(float_add "$start_time" "$CHUNK_DURATION")
        ((chunk_index++))
    done
    
    # Clean up temp file if created
    if [[ -n "$temp_mp4" ]] && [[ -f "$temp_mp4" ]]; then
        rm -f "$temp_mp4"
    fi
    
    return 0
}

# ============================================================================
# Main entry point
# ============================================================================
main() {
    # Parse arguments
    parse_args "$@"
    
    # Check dependencies
    check_dependencies
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    log_info "Starting video splitting process"
    log_info "  Input directory:  $INPUT_DIR"
    log_info "  Output directory: $OUTPUT_DIR"
    log_info "  Chunk duration:   ${CHUNK_DURATION}s"
    echo ""
    
    # Count total files
    local total_files
    total_files=$(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mkv" -o -iname "*.mov" -o -iname "*.wmv" -o -iname "*.flv" -o -iname "*.webm" \) | wc -l)
    
    if [[ "$total_files" -eq 0 ]]; then
        log_warn "No video files found in: $INPUT_DIR"
        exit 0
    fi
    
    log_info "Found $total_files video file(s) to process"
    echo ""
    
    local current_file=0
    local processed=0
    local skipped=0
    
    # Process each video file
    while IFS= read -r -d '' file; do
        ((current_file++))
        
        local filename
        filename=$(basename -- "$file")

        # Skip files that look like previously generated chunks (e.g. xxx_chunk_001.mp4)
        if [[ "$filename" =~ _chunk_[0-9]{3}\.mp4$ ]]; then
            continue
        fi

        # Show progress
        local progress=$((current_file * 100 / total_files))
        log_progress "Processing ($current_file/$total_files) $progress%: $filename"
        
        if process_video "$file"; then
            ((processed++))
        else
            ((skipped++))
        fi
        
    done < <(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mkv" -o -iname "*.mov" -o -iname "*.wmv" -o -iname "*.flv" -o -iname "*.webm" \) -print0)

    # Clear progress line and show summary
    echo ""
    echo ""
    log_info "Processing complete!"
    log_info "  Total files:     $total_files"
    log_info "  Processed:       $processed"
    log_info "  Skipped:         $skipped"
    log_info "  Output location: $OUTPUT_DIR"
}

# Run main function
main "$@"
