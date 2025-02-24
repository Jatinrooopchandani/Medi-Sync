"""Secure model audit trail utilities"""
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import threading
from dataclasses import dataclass, asdict
import os

logger = logging.getLogger(__name__)

@dataclass
class AuditEvent:
    """Structured audit event data"""
    timestamp: str
    event_type: str
    model_type: str
    model_version: str
    operation: str
    status: str
    details: Dict[str, Any]
    hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return asdict(self)

class ModelAuditTrail:
    """Thread-safe model audit trail manager"""
    
    def __init__(self, audit_dir: str, max_file_size: int = 10 * 1024 * 1024):
        """Initialize audit trail manager.
        
        Args:
            audit_dir: Directory for audit logs
            max_file_size: Maximum size of single audit file in bytes (default: 10MB)
        """
        self._lock = threading.Lock()
        self.audit_dir = Path(audit_dir)
        self.max_file_size = max_file_size
        self.current_file: Optional[Path] = None
        
        # Create audit directory if it doesn't exist
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        # Secure directory permissions
        os.chmod(self.audit_dir, 0o750)
        
        # Initialize current audit file
        self._initialize_audit_file()
    
    def _initialize_audit_file(self):
        """Initialize or rotate audit file if needed"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        self.current_file = self.audit_dir / f"model_audit_{timestamp}.json"
        
        if not self.current_file.exists():
            with open(self.current_file, 'w') as f:
                json.dump([], f)
            # Secure file permissions
            os.chmod(self.current_file, 0o640)
    
    def _rotate_audit_file(self):
        """Rotate audit file if it exceeds max size"""
        if self.current_file and self.current_file.stat().st_size >= self.max_file_size:
            self._initialize_audit_file()
    
    def _compute_event_hash(self, event: Dict[str, Any]) -> str:
        """Compute cryptographic hash of audit event"""
        # Create deterministic string representation
        event_str = json.dumps(event, sort_keys=True)
        return hashlib.sha256(event_str.encode()).hexdigest()
    
    def record_event(self, 
                    event_type: str,
                    model_type: str,
                    model_version: str,
                    operation: str,
                    status: str,
                    details: Dict[str, Any]) -> str:
        """Record an audit event.
        
        Args:
            event_type: Type of event (e.g., 'model_load', 'prediction')
            model_type: Type of model involved
            model_version: Version of model
            operation: Operation performed
            status: Status of operation
            details: Additional event details
            
        Returns:
            str: Hash of recorded event
            
        Raises:
            IOError: If audit file operations fail
        """
        with self._lock:
            try:
                # Rotate file if needed
                self._rotate_audit_file()
                
                # Create event
                event = AuditEvent(
                    timestamp=datetime.utcnow().isoformat(),
                    event_type=event_type,
                    model_type=model_type,
                    model_version=model_version,
                    operation=operation,
                    status=status,
                    details=details
                )
                
                # Compute event hash
                event_dict = event.to_dict()
                event_hash = self._compute_event_hash(event_dict)
                event.hash = event_hash
                
                # Read current audit file
                with open(self.current_file, 'r') as f:
                    audit_log = json.load(f)
                
                # Append new event
                audit_log.append(event.to_dict())
                
                # Write updated log
                with open(self.current_file, 'w') as f:
                    json.dump(audit_log, f, indent=2)
                
                logger.debug(f"Recorded audit event: {event_hash}")
                return event_hash
                
            except Exception as e:
                logger.error(f"Failed to record audit event: {str(e)}")
                raise
    
    def query_events(self,
                    start_time: Optional[str] = None,
                    end_time: Optional[str] = None,
                    event_type: Optional[str] = None,
                    model_type: Optional[str] = None,
                    status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query audit events with filters.
        
        Args:
            start_time: ISO format timestamp for start of range
            end_time: ISO format timestamp for end of range
            event_type: Filter by event type
            model_type: Filter by model type
            status: Filter by status
            
        Returns:
            List of matching audit events
        """
        with self._lock:
            matching_events = []
            
            # Convert timestamps if provided
            start_dt = datetime.fromisoformat(start_time) if start_time else None
            end_dt = datetime.fromisoformat(end_time) if end_time else None
            
            # Collect events from all audit files
            for audit_file in sorted(self.audit_dir.glob("model_audit_*.json")):
                try:
                    with open(audit_file, 'r') as f:
                        events = json.load(f)
                        
                    for event in events:
                        # Apply filters
                        event_dt = datetime.fromisoformat(event['timestamp'])
                        
                        if start_dt and event_dt < start_dt:
                            continue
                        if end_dt and event_dt > end_dt:
                            continue
                        if event_type and event['event_type'] != event_type:
                            continue
                        if model_type and event['model_type'] != model_type:
                            continue
                        if status and event['status'] != status:
                            continue
                            
                        matching_events.append(event)
                        
                except Exception as e:
                    logger.error(f"Error reading audit file {audit_file}: {str(e)}")
            
            return matching_events
    
    def verify_event_integrity(self, event_hash: str) -> bool:
        """Verify the integrity of a recorded event.
        
        Args:
            event_hash: Hash of event to verify
            
        Returns:
            bool: True if event is found and hash verifies
        """
        with self._lock:
            for audit_file in self.audit_dir.glob("model_audit_*.json"):
                try:
                    with open(audit_file, 'r') as f:
                        events = json.load(f)
                        
                    for event in events:
                        if event.get('hash') == event_hash:
                            # Verify hash
                            event_copy = event.copy()
                            event_copy.pop('hash')  # Remove hash for verification
                            computed_hash = self._compute_event_hash(event_copy)
                            return computed_hash == event_hash
                            
                except Exception as e:
                    logger.error(f"Error verifying event in {audit_file}: {str(e)}")
            
            return False
    
    def get_recent_activity(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent audit events.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of recent audit events
        """
        events = self.query_events()
        return sorted(events, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def get_model_history(self, model_type: str) -> List[Dict[str, Any]]:
        """Get complete history for a specific model type.
        
        Args:
            model_type: Type of model to query
            
        Returns:
            List of audit events for the model
        """
        return self.query_events(model_type=model_type)