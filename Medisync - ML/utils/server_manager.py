"""Secure server management utilities"""
import os
import sys
import signal
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
from waitress import create_server
from pathlib import Path
import json
import threading

logger = logging.getLogger(__name__)

class ServerManager:
    """Manage server lifecycle with security and monitoring"""
    
    def __init__(self, app, config: Dict[str, Any], audit_trail: Any, metrics_collector: Any):
        """Initialize server manager.
        
        Args:
            app: Flask application instance
            config: Server configuration dictionary
            audit_trail: Audit trail instance
            metrics_collector: Metrics collector instance
        """
        self.app = app
        self.config = config
        self.audit_trail = audit_trail
        self.metrics_collector = metrics_collector
        self.server = None
        self.start_time = None
        self._shutdown_event = threading.Event()
        self._connections_lock = threading.Lock()
        self.active_connections = set()
    
    def track_connection(self, conn):
        """Track a new connection"""
        with self._connections_lock:
            self.active_connections.add(conn)
    
    def untrack_connection(self, conn):
        """Remove a tracked connection"""
        with self._connections_lock:
            self.active_connections.discard(conn)
    
    def get_connection_count(self) -> int:
        """Get current number of active connections"""
        with self._connections_lock:
            return len(self.active_connections)
    
    def start(self):
        """Start the server with security measures"""
        try:
            # Record startup attempt
            self.audit_trail.record_event(
                event_type='server_start',
                model_type='system',
                model_version='current',
                operation='initialize',
                status='in_progress',
                details={
                    'config': {k: v for k, v in self.config.items() 
                             if k not in ['trusted_proxy', 'ssl_options', 'ciphers']},
                    'pid': os.getpid()
                }
            )
            
            # Create server instance
            self.server = create_server(
                self.app,
                host=self.config['host'],
                port=self.config['port'],
                **{k: v for k, v in self.config.items() 
                   if k not in ['host', 'port']}
            )
            
            # Set up connection tracking
            def on_connect(conn):
                self.track_connection(conn)
                
            def on_disconnect(conn):
                self.untrack_connection(conn)
            
            self.server.connection_made = on_connect
            self.server.connection_lost = on_disconnect
            
            # Record successful startup
            self.start_time = datetime.utcnow()
            self.audit_trail.record_event(
                event_type='server_start',
                model_type='system',
                model_version='current',
                operation='complete',
                status='success',
                details={
                    'start_time': self.start_time.isoformat(),
                    'host': self.config['host'],
                    'port': self.config['port']
                }
            )
            
            logger.info(f"Server starting on {self.config['host']}:{self.config['port']}")
            return self.server
            
        except Exception as e:
            logger.critical(f"Failed to start server: {str(e)}")
            self.audit_trail.record_event(
                event_type='server_start',
                model_type='system',
                model_version='current',
                operation='error',
                status='failed',
                details={'error': str(e)}
            )
            raise
    
    def initiate_shutdown(self):
        """Initiate graceful shutdown sequence"""
        logger.info("Initiating graceful shutdown...")
        shutdown_start = time.time()
        
        try:
            # Record shutdown initiation
            self.audit_trail.record_event(
                event_type='server_shutdown',
                model_type='system',
                model_version='current',
                operation='initiate',
                status='in_progress',
                details={
                    'active_connections': self.get_connection_count(),
                    'uptime': str(datetime.utcnow() - self.start_time) if self.start_time else 'unknown'
                }
            )
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Stop accepting new connections
            if self.server:
                self.server.active = False
            
            # Wait for existing connections
            wait_start = time.time()
            while (self.get_connection_count() > 0 and 
                   time.time() - wait_start < self.config.get('graceful_timeout', 30)):
                logger.info(f"Waiting for {self.get_connection_count()} connections to complete...")
                time.sleep(1)
            
            # Force close remaining connections
            remaining = self.get_connection_count()
            if remaining > 0:
                logger.warning(f"Forcing close of {remaining} connections")
                with self._connections_lock:
                    for conn in self.active_connections:
                        try:
                            conn.close()
                        except Exception as e:
                            logger.error(f"Error closing connection: {str(e)}")
            
            # Record successful shutdown
            shutdown_duration = time.time() - shutdown_start
            self.audit_trail.record_event(
                event_type='server_shutdown',
                model_type='system',
                model_version='current',
                operation='complete',
                status='success',
                details={
                    'duration': f"{shutdown_duration:.2f}s",
                    'forced_connections': remaining
                }
            )
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            self.audit_trail.record_event(
                event_type='server_shutdown',
                model_type='system',
                model_version='current',
                operation='error',
                status='failed',
                details={'error': str(e)}
            )
            raise
    
    def is_shutting_down(self) -> bool:
        """Check if server is in shutdown state"""
        return self._shutdown_event.is_set()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current server status"""
        status = {
            'status': 'running' if not self.is_shutting_down() else 'shutting_down',
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime': str(datetime.utcnow() - self.start_time) if self.start_time else None,
            'active_connections': self.get_connection_count(),
            'host': self.config['host'],
            'port': self.config['port']
        }
        
        if self.metrics_collector:
            try:
                status['metrics'] = self.metrics_collector.get_security_report()
            except Exception as e:
                logger.error(f"Error getting metrics: {str(e)}")
                status['metrics'] = {'error': str(e)}
        
        return status