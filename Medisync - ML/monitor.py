#!/usr/bin/env python3
"""
Production monitoring script for Medisync ML application.
Integrates with external monitoring systems and provides alerting capabilities.
"""
import os
import sys
import json
import time
import logging
import smtplib
import requests
import psutil
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/monitoring.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MedisyncMonitor:
    def __init__(self, config_path='monitoring_config.json'):
        self.load_config(config_path)
        self.alerts = []
        self.create_directories()

    def load_config(self, config_path):
        """Load monitoring configuration"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            sys.exit(1)

    def create_directories(self):
        """Ensure all required directories exist"""
        for path in self.config['paths'].values():
            Path(path).mkdir(parents=True, exist_ok=True)

    def check_system_health(self):
        """Perform comprehensive system health check"""
        health_data = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'checks': {},
            'alerts': []
        }

        # Check disk space
        health_data['checks']['disk'] = self.check_disk_space()

        # Check memory usage
        health_data['checks']['memory'] = self.check_memory_usage()

        # Check log files
        health_data['checks']['logs'] = self.check_log_files()

        # Check model files
        health_data['checks']['models'] = self.check_model_files()

        # Check application health endpoint
        health_data['checks']['application'] = self.check_application_health()

        # Update overall status if any checks failed
        if any(not check['status'] for check in health_data['checks'].values()):
            health_data['status'] = 'degraded'

        # Add any alerts that were generated
        health_data['alerts'] = self.alerts

        return health_data

    def check_disk_space(self):
        """Check disk space usage"""
        try:
            disk = psutil.disk_usage('.')
            free_gb = disk.free / (1024 ** 3)
            
            result = {
                'status': True,
                'free_gb': round(free_gb, 2),
                'percent_used': disk.percent
            }

            if free_gb < self.config['thresholds']['disk_space']['critical_gb']:
                self.add_alert('CRITICAL', f'Critical disk space: {free_gb:.2f}GB free')
                result['status'] = False
            elif free_gb < self.config['thresholds']['disk_space']['warning_gb']:
                self.add_alert('WARNING', f'Low disk space: {free_gb:.2f}GB free')
                
            return result
        except Exception as e:
            logger.error(f"Disk check failed: {str(e)}")
            return {'status': False, 'error': str(e)}

    def check_memory_usage(self):
        """Check system memory usage"""
        try:
            memory = psutil.virtual_memory()
            result = {
                'status': True,
                'percent_used': memory.percent,
                'available_gb': round(memory.available / (1024 ** 3), 2)
            }

            if memory.percent > self.config['thresholds']['memory_usage']['critical_percent']:
                self.add_alert('CRITICAL', f'Critical memory usage: {memory.percent}%')
                result['status'] = False
            elif memory.percent > self.config['thresholds']['memory_usage']['warning_percent']:
                self.add_alert('WARNING', f'High memory usage: {memory.percent}%')

            return result
        except Exception as e:
            logger.error(f"Memory check failed: {str(e)}")
            return {'status': False, 'error': str(e)}

    def check_log_files(self):
        """Check log files status"""
        try:
            log_dir = self.config['paths']['logs']
            result = {
                'status': True,
                'files_checked': 0,
                'issues': []
            }

            for log_file in Path(log_dir).glob('*.log'):
                size_mb = log_file.stat().st_size / (1024 * 1024)
                if size_mb > self.config['thresholds']['log_size']['max_mb']:
                    msg = f'Log file too large: {log_file.name} ({size_mb:.2f}MB)'
                    result['issues'].append(msg)
                    self.add_alert('WARNING', msg)
                    result['status'] = False
                result['files_checked'] += 1

            return result
        except Exception as e:
            logger.error(f"Log check failed: {str(e)}")
            return {'status': False, 'error': str(e)}

    def check_model_files(self):
        """Check ML model files"""
        try:
            model_dir = self.config['paths']['models']
            result = {
                'status': True,
                'files_checked': 0,
                'issues': []
            }

            for model_file in Path(model_dir).glob('*.*'):
                # Check file age
                age_days = (datetime.now() - datetime.fromtimestamp(model_file.stat().st_mtime)).days
                if age_days > self.config['thresholds']['model_age']['critical_days']:
                    msg = f'Model file too old: {model_file.name} ({age_days} days)'
                    result['issues'].append(msg)
                    self.add_alert('CRITICAL', msg)
                    result['status'] = False
                elif age_days > self.config['thresholds']['model_age']['warning_days']:
                    msg = f'Model file age warning: {model_file.name} ({age_days} days)'
                    result['issues'].append(msg)
                    self.add_alert('WARNING', msg)
                result['files_checked'] += 1

            return result
        except Exception as e:
            logger.error(f"Model check failed: {str(e)}")
            return {'status': False, 'error': str(e)}

    def check_application_health(self):
        """Check application health endpoint"""
        try:
            timeout = self.config['health_check']['timeout_seconds']
            response = requests.get(
                f"http://localhost:8080{self.config['health_check']['endpoint']}",
                timeout=timeout
            )
            
            result = {
                'status': True,
                'response_time_ms': round(response.elapsed.total_seconds() * 1000, 2),
                'status_code': response.status_code
            }

            if response.status_code != self.config['health_check']['expected_status']:
                msg = f'Health check failed: status {response.status_code}'
                self.add_alert('CRITICAL', msg)
                result['status'] = False

            return result
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {'status': False, 'error': str(e)}

    def add_alert(self, level, message):
        """Add an alert to the current monitoring run"""
        alert = {
            'level': level,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.alerts.append(alert)
        logger.warning(f"{level}: {message}")

    def send_alerts(self, health_data):
        """Send alerts through configured channels"""
        if not self.alerts:
            return

        if self.config['monitoring']['alert_channels']['email']['enabled']:
            self.send_email_alert(health_data)

        if self.config['monitoring']['alert_channels']['slack']['enabled']:
            self.send_slack_alert(health_data)

    def send_email_alert(self, health_data):
        """Send email alerts"""
        try:
            # Implementation would depend on your email configuration
            logger.info("Would send email alert here")
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")

    def send_slack_alert(self, health_data):
        """Send Slack alerts"""
        try:
            # Implementation would depend on your Slack configuration
            logger.info("Would send Slack alert here")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")

    def save_report(self, health_data):
        """Save monitoring report to file"""
        try:
            report_path = Path(self.config['paths']['reports'])
            filename = f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_path / filename, 'w') as f:
                json.dump(health_data, f, indent=2)
            
            logger.info(f"Report saved: {filename}")
        except Exception as e:
            logger.error(f"Failed to save report: {str(e)}")

    def cleanup_old_reports(self):
        """Clean up old monitoring reports"""
        try:
            report_path = Path(self.config['paths']['reports'])
            retention_days = self.config['monitoring']['report_retention_days']
            cutoff = datetime.now() - timedelta(days=retention_days)

            for report in report_path.glob('health_report_*.json'):
                if datetime.fromtimestamp(report.stat().st_mtime) < cutoff:
                    report.unlink()
        except Exception as e:
            logger.error(f"Failed to cleanup reports: {str(e)}")

def main():
    monitor = MedisyncMonitor()
    health_data = monitor.check_system_health()
    
    # Save report
    monitor.save_report(health_data)
    
    # Send alerts if needed
    monitor.send_alerts(health_data)
    
    # Cleanup old reports
    monitor.cleanup_old_reports()
    
    # Exit with status code
    sys.exit(0 if health_data['status'] == 'healthy' else 1)

if __name__ == '__main__':
    main()