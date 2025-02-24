"""Secure server startup and monitoring"""
import os
import sys
import threading
import time
from datetime import datetime
import logging
import psutil
from pathlib import Path
from typing import Optional

from main import (
    app, audit_trail, metrics_collector, manifest_manager,
    health_monitor, initialize_server, configure_server,
    cleanup_prediction_resources, shutdown_security_components
)

logger = logging.getLogger(__name__)

class ServerMonitor:
    """Monitor server health and resources"""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self._monitoring = False
        self._threads = []
    
    def start_verification_thread(self):
        """Start model verification monitoring"""
        def model_verification_loop():
            while self._monitoring:
                try:
                    manifest_manager.verify_integrity()
                    time.sleep(3600)  # Check every hour
                except Exception as e:
                    logger.error(f"Error in model verification: {str(e)}")
                    time.sleep(300)  # Retry after 5 minutes on error
        
        thread = threading.Thread(
            target=model_verification_loop,
            daemon=True,
            name="ModelVerification"
        )
        self._threads.append(thread)
        thread.start()
    
    def start_metrics_thread(self):
        """Start metrics collection"""
        def metrics_loop():
            while self._monitoring:
                try:
                    process = psutil.Process()
                    metrics_collector.record_system_metrics(
                        memory_usage=process.memory_percent(),
                        upload_size=sum(
                            f.stat().st_size for f in Path(app.config['UPLOAD_FOLDER']).glob('*')
                            if f.is_file()
                        )
                    )
                    time.sleep(self.check_interval)
                except Exception as e:
                    logger.error(f"Error collecting metrics: {str(e)}")
                    time.sleep(30)
        
        thread = threading.Thread(
            target=metrics_loop,
            daemon=True,
            name="MetricsCollection"
        )
        self._threads.append(thread)
        thread.start()
    
    def start_resource_monitor_thread(self):
        """Start resource usage monitoring"""
        def resource_monitor_loop():
            while self._monitoring:
                try:
                    process = psutil.Process()
                    
                    # Check memory usage
                    mem_percent = process.memory_percent()
                    if mem_percent > 90:
                        logger.warning(f"High memory usage detected: {mem_percent}%")
                        audit_trail.record_event(
                            event_type='resource_warning',
                            model_type='system',
                            model_version='current',
                            operation='monitor',
                            status='warning',
                            details={'memory_usage': mem_percent}
                        )
                    
                    # Check CPU usage
                    cpu_percent = process.cpu_percent()
                    if cpu_percent > 80:
                        logger.warning(f"High CPU usage detected: {cpu_percent}%")
                        audit_trail.record_event(
                            event_type='resource_warning',
                            model_type='system',
                            model_version='current',
                            operation='monitor',
                            status='warning',
                            details={'cpu_usage': cpu_percent}
                        )
                    
                    # Check open files
                    open_files = len(process.open_files())
                    if open_files > 900:
                        logger.warning(f"High number of open files: {open_files}")
                        audit_trail.record_event(
                            event_type='resource_warning',
                            model_type='system',
                            model_version='current',
                            operation='monitor',
                            status='warning',
                            details={'open_files': open_files}
                        )
                    
                    time.sleep(self.check_interval)
                    
                except Exception as e:
                    logger.error(f"Error monitoring resources: {str(e)}")
                    time.sleep(30)
        
        thread = threading.Thread(
            target=resource_monitor_loop,
            daemon=True,
            name="ResourceMonitor"
        )
        self._threads.append(thread)
        thread.start()
    
    def start(self):
        """Start all monitoring threads"""
        self._monitoring = True
        
        # Start health monitor
        health_monitor.start()
        
        # Start verification thread
        self.start_verification_thread()
        
        # Start metrics thread
        self.start_metrics_thread()
        
        # Start resource monitor
        self.start_resource_monitor_thread()
        
        logger.info("All monitoring threads started successfully")
    
    def stop(self):
        """Stop all monitoring threads"""
        self._monitoring = False
        for thread in self._threads:
            try:
                thread.join(timeout=5)
            except Exception as e:
                logger.error(f"Error stopping thread {thread.name}: {str(e)}")
        self._threads = []
        logger.info("All monitoring threads stopped")

def run_server():
    """Run the server with full monitoring and security"""
    server_monitor = None
    
    try:
        # Initialize server components
        if not initialize_server():
            logger.critical("Server initialization failed")
            sys.exit(1)
        
        # Start monitoring
        server_monitor = ServerMonitor()
        server_monitor.start()
        
        # Configure and start server
        server_manager = configure_server()
        if not server_manager:
            raise RuntimeError("Server configuration failed")
        
        # Record startup
        audit_trail.record_event(
            event_type='server_startup',
            model_type='system',
            model_version='current',
            operation='complete',
            status='success',
            details={
                'pid': os.getpid(),
                'start_time': datetime.utcnow().isoformat(),
                'host': server_manager.config['host'],
                'port': server_manager.config['port']
            }
        )
        
        # Start server
        server = server_manager.start()
        server.run()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.critical(f"Critical server error: {str(e)}")
        audit_trail.record_event(
            event_type='server_error',
            model_type='system',
            model_version='current',
            operation='run',
            status='failed',
            details={'error': str(e)}
        )
    finally:
        # Clean shutdown
        try:
            if server_monitor:
                server_monitor.stop()
            cleanup_prediction_resources(None, None)
            shutdown_security_components()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    run_server()