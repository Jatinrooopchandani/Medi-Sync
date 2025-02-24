#!/usr/bin/env python3
"""
System health check script for Medisync ML application.
Performs comprehensive system checks and reports status.
"""
import os
import sys
import psutil
import requests
import logging
import json
from datetime import datetime
import shutil
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class SystemChecker:
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.issues = []

    def check_disk_space(self, min_free_gb=1):
        """Check if there's enough disk space"""
        try:
            total, used, free = shutil.disk_usage('.')
            free_gb = free // (2**30)
            if free_gb < min_free_gb:
                self.record_issue(f"Low disk space: {free_gb}GB free")
            else:
                self.checks_passed += 1
        except Exception as e:
            self.record_issue(f"Disk space check failed: {str(e)}")

    def check_memory_usage(self, max_percent=90):
        """Check system memory usage"""
        try:
            memory = psutil.virtual_memory()
            if memory.percent > max_percent:
                self.record_issue(f"High memory usage: {memory.percent}%")
            else:
                self.checks_passed += 1
        except Exception as e:
            self.record_issue(f"Memory check failed: {str(e)}")

    def check_required_files(self):
        """Check if all required files exist and are readable"""
        required_files = [
            'main.py',
            'requirements.txt',
            'datasets/symtoms_df.csv',
            'datasets/precautions_df.csv',
            'datasets/workout_df.csv',
            'datasets/description.csv',
            'datasets/medications.csv',
            'datasets/diets.csv',
            'models/rf.pkl',
            'models/lstm_drug_model.keras',
            'models/drug_encoder.pkl',
            'models/tokenizer.pkl'
        ]
        
        for file in required_files:
            if not os.path.isfile(file):
                self.record_issue(f"Missing required file: {file}")
            elif not os.access(file, os.R_OK):
                self.record_issue(f"File not readable: {file}")
            else:
                self.checks_passed += 1

    def check_log_rotation(self):
        """Check if log rotation is working"""
        log_dir = 'logs'
        archive_dir = os.path.join(log_dir, 'archive')
        
        if not os.path.exists(archive_dir):
            self.record_issue("Log archive directory missing")
            return
        
        try:
            # Check if archived logs are not too old
            archives = os.listdir(archive_dir)
            if archives:
                latest_log = max(archives)
                log_date = datetime.strptime(latest_log.split('_')[1].split('.')[0], '%Y%m%d%H%M%S')
                days_old = (datetime.now() - log_date).days
                
                if days_old > 7:  # Warning if latest archive is more than 7 days old
                    self.record_issue(f"Latest log archive is {days_old} days old")
            self.checks_passed += 1
        except Exception as e:
            self.record_issue(f"Log rotation check failed: {str(e)}")

    def check_model_files(self):
        """Verify ML model files integrity"""
        try:
            # Check if model files are readable and non-empty
            model_files = [
                'models/rf.pkl',
                'models/lstm_drug_model.keras',
                'models/drug_encoder.pkl',
                'models/tokenizer.pkl'
            ]
            
            for model_file in model_files:
                if not os.path.exists(model_file):
                    self.record_issue(f"Model file missing: {model_file}")
                    continue
                    
                if os.path.getsize(model_file) == 0:
                    self.record_issue(f"Empty model file: {model_file}")
                    continue
                
                self.checks_passed += 1
        except Exception as e:
            self.record_issue(f"Model files check failed: {str(e)}")

    def check_dataset_integrity(self):
        """Verify dataset files integrity"""
        try:
            dataset_files = [
                'datasets/symtoms_df.csv',
                'datasets/precautions_df.csv',
                'datasets/workout_df.csv',
                'datasets/description.csv',
                'datasets/medications.csv',
                'datasets/diets.csv'
            ]
            
            for dataset in dataset_files:
                if not os.path.exists(dataset):
                    self.record_issue(f"Dataset missing: {dataset}")
                    continue
                
                # Try reading the CSV file
                try:
                    df = pd.read_csv(dataset)
                    if df.empty:
                        self.record_issue(f"Empty dataset: {dataset}")
                    else:
                        self.checks_passed += 1
                except Exception as e:
                    self.record_issue(f"Failed to read dataset {dataset}: {str(e)}")
        except Exception as e:
            self.record_issue(f"Dataset integrity check failed: {str(e)}")

    def record_issue(self, issue):
        """Record a failed check"""
        self.checks_failed += 1
        self.issues.append(issue)
        logger.warning(issue)

    def run_checks(self):
        """Run all system checks"""
        logger.info("Starting system checks...")
        
        self.check_disk_space()
        self.check_memory_usage()
        self.check_required_files()
        self.check_log_rotation()
        self.check_model_files()
        self.check_dataset_integrity()
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'checks_passed': self.checks_passed,
            'checks_failed': self.checks_failed,
            'issues': self.issues,
            'status': 'healthy' if self.checks_failed == 0 else 'degraded'
        }
        
        # Save report
        report_file = f'logs/system_check_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {report_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {str(e)}")
        
        return report

if __name__ == '__main__':
    checker = SystemChecker()
    report = checker.run_checks()
    
    print("\nSystem Check Report:")
    print(f"Status: {report['status']}")
    print(f"Checks Passed: {report['checks_passed']}")
    print(f"Checks Failed: {report['checks_failed']}")
    
    if report['issues']:
        print("\nIssues Found:")
        for issue in report['issues']:
            print(f"- {issue}")
            
    sys.exit(1 if report['checks_failed'] > 0 else 0)