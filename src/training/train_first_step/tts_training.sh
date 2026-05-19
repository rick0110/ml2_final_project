#!/bin/bash
# TTS Training Helper Script
# Quick access to common training commands

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
TRAINING_DIR="$PROJECT_ROOT/src/training/train_first_step"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Command: help
cmd_help() {
    cat << EOF

TTS Training Helper Script

Usage: ./tts_training.sh <command> [options]

Commands:
  test              Run setup tests
  quick-test        Quick training test (5 epochs)
  balanced          Balanced training (100 epochs) [DEFAULT]
  production        Production training (200 epochs)
  high-diversity    High diversity emphasis
  lightweight       Lightweight (limited resources)
  
  list-configs      List all configurations
  list-experiments  List all experiments
  inspect           Inspect a checkpoint
  
  tensorboard       Start TensorBoard (interactive)
  clean             Remove experiment files
  help              Show this help message

Examples:
  ./tts_training.sh test                  # Verify setup
  ./tts_training.sh balanced              # Start balanced training
  ./tts_training.sh tensorboard           # View TensorBoard
  ./tts_training.sh list-experiments      # List your experiments

EOF
}

# Command: test
cmd_test() {
    print_header "Testing Setup"
    python "$TRAINING_DIR/test_setup.py"
    print_success "Setup test completed"
}

# Command: quick-test
cmd_quick_test() {
    print_header "Quick Test Training (5 epochs)"
    print_info "This will train for ~5-10 minutes"
    python "$TRAINING_DIR/run_training.py" quick_test
}

# Command: balanced
cmd_balanced() {
    print_header "Balanced Training (100 epochs)"
    print_info "This will train for ~2-4 hours"
    print_warning "Make sure you have a GPU available"
    python "$TRAINING_DIR/run_training.py" balanced
}

# Command: production
cmd_production() {
    print_header "Production Training (200 epochs)"
    print_info "This will train for ~8-12 hours"
    print_warning "Requires high-end GPU with significant memory"
    python "$TRAINING_DIR/run_training.py" production
}

# Command: high-diversity
cmd_high_diversity() {
    print_header "High Diversity Training (100 epochs)"
    print_info "Emphasizes style variation"
    python "$TRAINING_DIR/run_training.py" high_diversity
}

# Command: lightweight
cmd_lightweight() {
    print_header "Lightweight Training (50 epochs)"
    print_info "For limited GPU resources"
    python "$TRAINING_DIR/run_training.py" lightweight
}

# Command: list-configs
cmd_list_configs() {
    print_header "Available Configurations"
    python "$TRAINING_DIR/configs.py"
}

# Command: list-experiments
cmd_list_experiments() {
    print_header "Your Experiments"
    python "$TRAINING_DIR/checkpoint_utils.py" list-experiments
}

# Command: inspect
cmd_inspect() {
    if [ -z "$1" ]; then
        echo "Usage: ./tts_training.sh inspect <path-to-checkpoint>"
        echo "Example: ./tts_training.sh inspect experiments/step_1/attempt_20240101_120000/checkpoints/best.pt"
        return 1
    fi
    
    print_header "Inspecting Checkpoint"
    python "$TRAINING_DIR/checkpoint_utils.py" inspect "$1"
}

# Command: tensorboard
cmd_tensorboard() {
    EXPERIMENT_DIR=$(find "$PROJECT_ROOT/experiments/step_1" -maxdepth 1 -type d -name "attempt_*" | sort -r | head -1)
    
    if [ -z "$EXPERIMENT_DIR" ]; then
        print_warning "No experiments found"
        return 1
    fi
    
    TB_DIR="$EXPERIMENT_DIR/tensorboard"
    
    if [ ! -d "$TB_DIR" ]; then
        print_warning "TensorBoard directory not found in $EXPERIMENT_DIR"
        return 1
    fi
    
    print_header "Starting TensorBoard"
    print_info "Latest experiment: $(basename "$EXPERIMENT_DIR")"
    print_info "View at: http://localhost:6006"
    print_info "Press Ctrl+C to stop"
    
    python -m tensorboard.main --logdir="$TB_DIR" --host=0.0.0.0 --port=6006
}

# Command: clean
cmd_clean() {
    print_warning "This will remove all experiment files!"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        print_info "Cleanup cancelled"
        return 0
    fi
    
    print_header "Cleaning Experiments"
    rm -rf "$PROJECT_ROOT/experiments/step_1"
    print_success "All experiments removed"
}

# Main script logic
main() {
    cd "$PROJECT_ROOT"
    
    if [ $# -eq 0 ]; then
        cmd_help
        return 0
    fi
    
    COMMAND="$1"
    shift
    
    case "$COMMAND" in
        help|-h|--help)
            cmd_help
            ;;
        test)
            cmd_test
            ;;
        quick-test)
            cmd_quick_test
            ;;
        balanced)
            cmd_balanced
            ;;
        production)
            cmd_production
            ;;
        high-diversity)
            cmd_high_diversity
            ;;
        lightweight)
            cmd_lightweight
            ;;
        list-configs)
            cmd_list_configs
            ;;
        list-experiments)
            cmd_list_experiments
            ;;
        inspect)
            cmd_inspect "$@"
            ;;
        tensorboard)
            cmd_tensorboard
            ;;
        clean)
            cmd_clean
            ;;
        *)
            echo -e "${YELLOW}Unknown command: $COMMAND${NC}"
            echo "Use './tts_training.sh help' for available commands"
            return 1
            ;;
    esac
}

# Run main
main "$@"
