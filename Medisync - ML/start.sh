#!/bin/bash

# Exit on any error
set -e

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to create backup
backup() {
    local backup_dir="backups"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="${backup_dir}/medisync_${timestamp}.tar.gz"
    
    mkdir -p "$backup_dir"
    
    log "Creating backup: $backup_file"
    tar -czf "$backup_file" \
        --exclude="*.pyc" \
        --exclude="__pycache__" \
        --exclude="logs/*" \
        --exclude="backups/*" \
        ./*
    
    # Keep only last 5 backups
    ls -t "${backup_dir}"/medisync_*.tar.gz | tail -n +6 | xargs -r rm
    
    log "Backup completed: $backup_file"
}

# Function to rotate logs
rotate_logs() {
    log "Rotating logs..."
    
    # Create archive directory if it doesn't exist
    mkdir -p logs/archive
    
    # Get timestamp for archive names
    timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Rotate main application log
    if [ -f "logs/medisync.log" ]; then
        mv "logs/medisync.log" "logs/archive/medisync_${timestamp}.log"
        touch "logs/medisync.log"
    fi
    
    # Rotate monitoring log
    if [ -f "logs/monitoring.log" ]; then
        mv "logs/monitoring.log" "logs/archive/monitoring_${timestamp}.log"
        touch "logs/monitoring.log"
    fi
    
    # Compress logs older than 1 day
    find logs/archive -type f -name "*.log" -mtime +1 -exec gzip {} \;
    
    # Remove logs older than 30 days
    find logs/archive -type f -name "*.log.gz" -mtime +30 -delete
    
    log "Log rotation completed"
}

# Function to check permissions
check_permissions() {
    log "Checking file permissions..."
    
    # Set correct permissions for directories
    chmod 755 . logs models datasets templates
    chmod 755 logs/archive
    
    # Set correct permissions for executable files
    chmod 755 start.sh system_check.py monitor.py
    
    # Set correct permissions for configuration files
    chmod 644 requirements.txt README.md main.py monitoring_config.json
    chmod 644 medisync.service medisync.crontab
    
    # Set correct permissions for log files
    chmod 644 logs/*.log
    chmod 644 logs/archive/*
    
    log "File permissions updated"
}

# Function to check Python version
check_python_version() {
    required_version="3.8"
    current_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [ "$(printf '%s\n' "$required_version" "$current_version" | sort -V | head -n1)" != "$required_version" ]; then
        log "ERROR: Python version $required_version or higher is required. Current version: $current_version"
        exit 1
    fi
}

log "Starting Medisync ML initialization..."

# Check Python version
log "Checking Python version..."
check_python_version

# Create necessary directories
log "Creating directory structure..."
mkdir -p logs
mkdir -p datasets
mkdir -p models
mkdir -p logs/archive  # For log rotation

# Check for required dataset and model files
log "Checking required files..."
required_files=(
    "datasets/symtoms_df.csv"
    "datasets/precautions_df.csv"
    "datasets/workout_df.csv"
    "datasets/description.csv"
    "datasets/medications.csv"
    "datasets/diets.csv"
    "models/rf.pkl"
    "models/lstm_drug_model.keras"
    "models/drug_encoder.pkl"
    "models/tokenizer.pkl"
)

missing_files=0
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        log "ERROR: Missing required file: $file"
        missing_files=1
    fi
done

if [ $missing_files -eq 1 ]; then
    log "Please ensure all required files are present before starting the application."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    log "Creating default .env configuration file..."
    cat > .env << EOF
# Flask Configuration
FLASK_APP=main.py
FLASK_ENV=production  # Change to 'development' for development mode
FLASK_DEBUG=False    # Change to 'True' for development mode

# Server Configuration
HOST=0.0.0.0
PORT=8080

# Logging Configuration
LOG_LEVEL=INFO

# Security Configuration
MAX_CONTENT_LENGTH=1048576  # 1MB in bytes
RATE_LIMIT_DEFAULT="200 per day"
EOF
    log "Created default .env configuration file"
fi

# Install or upgrade dependencies
log "Checking and installing dependencies..."
pip install -r requirements.txt --upgrade

# Run startup tests
log "Running startup tests..."
if ! python3 test_startup.py; then
    log "ERROR: Startup tests failed. Please check the logs for details."
    exit 1
fi

# Rotate old logs if they exist
if [ -f "logs/medisync.log" ]; then
    log "Rotating old logs..."
    timestamp=$(date +%Y%m%d_%H%M%S)
    mv logs/medisync.log "logs/archive/medisync_${timestamp}.log"
fi

# Start the application
log "All checks passed. Starting Medisync ML application..."
if [ "${FLASK_ENV:-production}" = "production" ]; then
    log "Running in PRODUCTION mode"
else
    log "Running in DEVELOPMENT mode"
fi

# Function to validate Python packages
check_dependencies() {
    log "Checking Python dependencies..."
    
    # Check if pip can import required packages
    required_packages=(
        "flask"
        "waitress"
        "numpy"
        "pandas"
        "tensorflow"
        "psutil"
        "requests"
    )
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" &>/dev/null; then
            log "ERROR: Required package not found: $package"
            return 1
        fi
    done
    
    log "All dependencies are satisfied"
    return 0
}

# Function to start the application
start_application() {
    log "Starting Medisync ML application..."
    
    # Run initial system check
    if ! ./system_check.py --quiet; then
        log "ERROR: System check failed. Please check logs for details."
        exit 1
    fi
    
    # Start the monitoring process in the background
    ./monitor.py --daemon &
    
    # Start the main application
    if [ "${FLASK_ENV:-production}" = "production" ]; then
        log "Running in PRODUCTION mode"
        exec python3 main.py
    else
        log "Running in DEVELOPMENT mode"
        FLASK_DEBUG=True exec python3 main.py
    fi
}

# Process command line arguments
case "${1:-start}" in
    start)
        check_permissions
        check_dependencies
        start_application
        ;;
    
    backup)
        backup
        ;;
    
    rotate-logs)
        rotate_logs
        ;;
    
    check-permissions)
        check_permissions
        ;;
    
    validate)
        check_permissions
        check_dependencies
        ./system_check.py
        ;;
    
    *)
        echo "Usage: $0 {start|backup|rotate-logs|check-permissions|validate}"
        exit 1
        ;;
esac