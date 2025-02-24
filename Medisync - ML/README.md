# Medisync ML Service

A secure machine learning service for medical diagnosis and recommendations.

## Security Improvements

This version includes several security enhancements:
- Disabled debug mode in production
- Replaced Flask development server with Waitress WSGI server
- Added comprehensive error handling and logging
- Implemented rate limiting and request validation
- Added security headers
- Added health monitoring endpoint
- Added proper environment configuration management

## Prerequisites

- Python 3.8 or higher
- Required data files in `datasets/` directory
- Required model files in `models/` directory
- Adequate disk space for logs and model files

## Directory Structure

```
Medisync - ML/
├── datasets/           # Dataset files
├── models/            # ML model files
├── logs/              # Application logs
│   └── archive/      # Archived logs
├── templates/         # HTML templates
├── main.py           # Main application
├── start.sh          # Startup script
├── test_startup.py   # Startup tests
├── requirements.txt  # Python dependencies
└── .env             # Environment configuration
```

## Installation

1. Clone the repository
2. Place required dataset and model files in their respective directories
3. Run the startup script:
   ```bash
   ./start.sh
   ```

The script will:
- Verify Python version
- Check for required files
- Install dependencies
- Create necessary directories
- Run startup tests
- Configure the environment
- Start the application

## Configuration

### Environment Variables

Configure the application using the `.env` file:

```ini
# Flask Configuration
FLASK_APP=main.py
FLASK_ENV=production
FLASK_DEBUG=False

# Server Configuration
HOST=0.0.0.0
PORT=8080

# Logging Configuration
LOG_LEVEL=INFO

# Security Configuration
MAX_CONTENT_LENGTH=1048576
RATE_LIMIT_DEFAULT="200 per day"
```

### Production Deployment

For production deployment:
1. Ensure `FLASK_ENV=production` and `FLASK_DEBUG=False`
2. Configure proper firewall rules
3. Set up HTTPS using a reverse proxy (e.g., Nginx)
4. Configure appropriate rate limits
5. Set up log rotation
6. Monitor the health endpoint regularly

### Health Monitoring

The `/health` endpoint provides application status:
```bash
curl http://localhost:8080/health
```

Response format:
```json
{
    "status": "healthy",
    "details": {
        "data_loaded": true,
        "models_loaded": true
    }
}
```

## Security Considerations

1. **Environment Configuration**
   - Never enable debug mode in production
   - Use environment variables for sensitive configuration
   - Regularly rotate any API keys or secrets

2. **Access Control**
   - Configure proper firewall rules
   - Use HTTPS in production
   - Monitor access logs

3. **Rate Limiting**
   - Default: 200 requests per day
   - Adjust based on your requirements

4. **Data Protection**
   - Ensure proper file permissions
   - Regular backup of logs and data
   - Sanitize all user inputs

5. **Monitoring**
   - Regular health checks
   - Log monitoring
   - Resource usage monitoring

## Error Handling

The application includes comprehensive error handling:
- Custom error pages
- Detailed logging
- Rate limit exceeded notifications
- Input validation errors
- Model prediction errors

## Logging

Logs are stored in `logs/medisync.log` with automatic rotation:
- Maximum file size: 10MB
- Keeps 10 backup files
- Archived logs in `logs/archive/`

## Troubleshooting

1. **Application won't start**
   - Check log files
   - Verify file permissions
   - Ensure all required files exist
   - Check Python version

2. **Performance issues**
   - Monitor resource usage
   - Check rate limiting configuration
   - Review access logs
   - Verify model loading times

3. **Prediction errors**
   - Check model files integrity
   - Verify input data format
   - Review error logs

## System Monitoring

### Automated System Checks

The application includes a comprehensive system monitoring script (`system_check.py`) that performs the following checks:
- Disk space usage
- Memory usage
- Required files presence and permissions
- Log rotation status
- ML model files integrity
- Dataset integrity
- System resource usage

To run a system check manually:
```bash
./system_check.py
```

### Automated Monitoring Setup

1. Create a monitoring crontab:
```bash
# Run system checks every hour
0 * * * * cd /path/to/Medisync-ML && ./system_check.py

# Rotate logs daily at midnight
0 0 * * * cd /path/to/Medisync-ML && ./start.sh rotate-logs

# Check disk space every 6 hours
0 */6 * * * cd /path/to/Medisync-ML && ./system_check.py --check disk-space
```

2. Monitor system check reports:
- Reports are stored in `logs/system_check_*.json`
- Each report includes:
  - Overall system status
  - Number of checks passed/failed
  - Detailed issue descriptions
  - Timestamp of the check

3. Set up alerts:
- Configure your monitoring system to watch for:
  - Failed system checks
  - High resource usage
  - Missing or corrupt files
  - Log rotation issues

### System Check Categories

1. **Resource Monitoring**
   - Disk space (warns at <1GB free)
   - Memory usage (warns at >90% usage)
   - Log file sizes

2. **File Integrity**
   - Required file presence
   - File permissions
   - Dataset integrity
   - Model file validation

3. **Log Management**
   - Log rotation status
   - Archive maintenance
   - Log file permissions

4. **Data Validation**
   - Dataset completeness
   - Model file integrity
   - Required directory structure

## Support

For security issues or bugs:
1. Check the logs
2. Run startup tests and system checks
3. Verify configuration
4. Review system check reports
5. Contact system administrator

## License

[Include your license information here]