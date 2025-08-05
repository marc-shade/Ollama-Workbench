# security/audit_logging.py

"""
Comprehensive Audit Logging System
Implements detailed security event logging, compliance tracking, and forensic capabilities.
"""

import os
import json
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import hashlib
import threading
from collections import deque
import streamlit as st
from .security_config import get_security_config

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Security event types"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_ACCESS = "system_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_INCIDENT = "security_incident"
    API_ACCESS = "api_access"
    FILE_ACCESS = "file_access"
    MODEL_ACCESS = "model_access"
    ERROR = "error"
    ADMIN_ACTION = "admin_action"

class EventSeverity(Enum):
    """Event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AuditEvent:
    """Represents a security audit event"""
    event_id: str
    timestamp: datetime
    event_type: EventType
    severity: EventSeverity
    user_id: str
    session_id: Optional[str]
    ip_address: str
    user_agent: Optional[str]
    resource: str
    action: str
    outcome: str  # success, failure, error
    details: Dict[str, Any]
    risk_score: int = 0  # 0-100
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['event_type'] = EventType(data['event_type'])
        data['severity'] = EventSeverity(data['severity'])
        return cls(**data)

class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self, db_path: str = "security/audit.db"):
        self.config = get_security_config()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Thread-safe event buffer
        self._event_buffer = deque(maxlen=10000)
        self._buffer_lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        # Start background processing
        self._start_background_processor()
    
    def _init_database(self) -> None:
        """Initialize audit database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS audit_events (
                        event_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        session_id TEXT,
                        ip_address TEXT NOT NULL,
                        user_agent TEXT,
                        resource TEXT NOT NULL,
                        action TEXT NOT NULL,
                        outcome TEXT NOT NULL,
                        details TEXT NOT NULL,
                        risk_score INTEGER DEFAULT 0,
                        tags TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_severity ON audit_events(severity)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_outcome ON audit_events(outcome)')
                
                # Create summary table for statistics
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS audit_statistics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        outcome TEXT NOT NULL,
                        count INTEGER DEFAULT 1,
                        UNIQUE(date, event_type, severity, outcome)
                    )
                ''')
                
                conn.commit()
                logger.info("Audit database initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize audit database: {e}")
    
    def _start_background_processor(self) -> None:
        """Start background thread for processing events"""
        def process_events():
            while True:
                try:
                    self._process_event_buffer()
                    self._cleanup_old_events()
                    threading.Event().wait(30)  # Process every 30 seconds
                except Exception as e:
                    logger.error(f"Background audit processing error: {e}")
                    threading.Event().wait(60)  # Back off on error
        
        processor_thread = threading.Thread(target=process_events, daemon=True)
        processor_thread.start()
        logger.info("Audit background processor started")
    
    def _process_event_buffer(self) -> None:
        """Process events from buffer to database"""
        events_to_process = []
        
        with self._buffer_lock:
            while self._event_buffer:
                events_to_process.append(self._event_buffer.popleft())
        
        if events_to_process:
            self._batch_insert_events(events_to_process)
    
    def _batch_insert_events(self, events: List[AuditEvent]) -> None:
        """Batch insert events to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                event_data = []
                stats_data = []
                
                for event in events:
                    # Prepare event data
                    event_data.append((
                        event.event_id,
                        event.timestamp.isoformat(),
                        event.event_type.value,
                        event.severity.value,
                        event.user_id,
                        event.session_id,
                        event.ip_address,
                        event.user_agent,
                        event.resource,
                        event.action,
                        event.outcome,
                        json.dumps(event.details),
                        event.risk_score,
                        json.dumps(event.tags)
                    ))
                    
                    # Prepare statistics data
                    date_str = event.timestamp.date().isoformat()
                    stats_data.append((
                        date_str,
                        event.event_type.value,
                        event.severity.value,
                        event.outcome
                    ))
                
                # Insert events
                conn.executemany('''
                    INSERT OR REPLACE INTO audit_events
                    (event_id, timestamp, event_type, severity, user_id, session_id,
                     ip_address, user_agent, resource, action, outcome, details, risk_score, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', event_data)
                
                # Update statistics
                for stats in stats_data:
                    conn.execute('''
                        INSERT INTO audit_statistics (date, event_type, severity, outcome, count)
                        VALUES (?, ?, ?, ?, 1)
                        ON CONFLICT(date, event_type, severity, outcome)
                        DO UPDATE SET count = count + 1
                    ''', stats)
                
                conn.commit()
                logger.debug(f"Processed {len(events)} audit events")
                
        except Exception as e:
            logger.error(f"Failed to batch insert audit events: {e}")
    
    def _cleanup_old_events(self) -> None:
        """Clean up old audit events based on retention policy"""
        if not self.config.audit_log_retention_days:
            return
        
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config.audit_log_retention_days)
            
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute(
                    'DELETE FROM audit_events WHERE timestamp < ?',
                    (cutoff_date.isoformat(),)
                )
                
                deleted_count = result.rowcount
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old audit events")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to cleanup old audit events: {e}")
    
    def log_event(self, event_type: EventType, severity: EventSeverity, user_id: str,
                  resource: str, action: str, outcome: str, details: Dict[str, Any] = None,
                  session_id: Optional[str] = None, ip_address: str = None,
                  user_agent: Optional[str] = None, tags: List[str] = None) -> str:
        """Log a security event"""
        
        if not self.config.enable_audit_logging:
            return ""
        
        # Generate event ID
        event_id = self._generate_event_id(user_id, resource, action)
        
        # Get context information
        if ip_address is None:
            ip_address = self._get_client_ip()
        
        if user_agent is None:
            user_agent = self._get_user_agent()
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(event_type, severity, outcome, details or {})
        
        # Create event
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            outcome=outcome,
            details=details or {},
            risk_score=risk_score,
            tags=tags or []
        )
        
        # Add to buffer for processing
        with self._buffer_lock:
            self._event_buffer.append(event)
        
        # Log to standard logger as well
        log_level = self._get_log_level(severity)
        logger.log(log_level, f"Audit: {event_type.value} - {user_id} {action} {resource} - {outcome}", 
                  extra=event.to_dict())
        
        return event_id
    
    def _generate_event_id(self, user_id: str, resource: str, action: str) -> str:
        """Generate unique event ID"""
        timestamp = datetime.now(timezone.utc).isoformat()
        content = f"{timestamp}:{user_id}:{resource}:{action}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _calculate_risk_score(self, event_type: EventType, severity: EventSeverity,
                             outcome: str, details: Dict[str, Any]) -> int:
        """Calculate risk score for event (0-100)"""
        base_score = 0
        
        # Base score from event type
        type_scores = {
            EventType.AUTHENTICATION: 30,
            EventType.AUTHORIZATION: 40,
            EventType.DATA_ACCESS: 20,
            EventType.DATA_MODIFICATION: 50,
            EventType.SYSTEM_ACCESS: 60,
            EventType.CONFIGURATION_CHANGE: 70,
            EventType.SECURITY_INCIDENT: 90,
            EventType.ADMIN_ACTION: 60,
            EventType.ERROR: 10
        }
        base_score = type_scores.get(event_type, 10)
        
        # Severity multiplier
        severity_multipliers = {
            EventSeverity.LOW: 0.5,
            EventSeverity.MEDIUM: 0.8,
            EventSeverity.HIGH: 1.2,
            EventSeverity.CRITICAL: 1.5
        }
        base_score *= severity_multipliers.get(severity, 1.0)
        
        # Outcome modifier
        if outcome == "failure":
            base_score *= 1.3
        elif outcome == "error":
            base_score *= 1.1
        
        # Additional risk factors from details
        if details.get("sensitive_data_accessed"):
            base_score *= 1.2
        if details.get("admin_privileges_used"):
            base_score *= 1.3
        if details.get("external_ip"):
            base_score *= 1.1
        
        return min(int(base_score), 100)
    
    def _get_log_level(self, severity: EventSeverity) -> int:
        """Get logging level from severity"""
        level_map = {
            EventSeverity.LOW: logging.INFO,
            EventSeverity.MEDIUM: logging.WARNING,
            EventSeverity.HIGH: logging.ERROR,
            EventSeverity.CRITICAL: logging.CRITICAL
        }
        return level_map.get(severity, logging.INFO)
    
    def _get_client_ip(self) -> str:
        """Get client IP address"""
        # In a real implementation, this would extract from request headers
        return "127.0.0.1"
    
    def _get_user_agent(self) -> Optional[str]:
        """Get user agent string"""
        # In a real implementation, this would extract from request headers
        return "Ollama-Workbench/1.0"
    
    def query_events(self, start_time: datetime = None, end_time: datetime = None,
                    event_types: List[EventType] = None, severities: List[EventSeverity] = None,
                    user_ids: List[str] = None, outcomes: List[str] = None,
                    limit: int = 1000, offset: int = 0) -> List[AuditEvent]:
        """Query audit events with filters"""
        
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        # Build query conditions
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        if event_types:
            placeholders = ",".join("?" * len(event_types))
            query += f" AND event_type IN ({placeholders})"
            params.extend([et.value for et in event_types])
        
        if severities:
            placeholders = ",".join("?" * len(severities))
            query += f" AND severity IN ({placeholders})"
            params.extend([s.value for s in severities])
        
        if user_ids:
            placeholders = ",".join("?" * len(user_ids))
            query += f" AND user_id IN ({placeholders})"
            params.extend(user_ids)
        
        if outcomes:
            placeholders = ",".join("?" * len(outcomes))
            query += f" AND outcome IN ({placeholders})"
            params.extend(outcomes)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                
                events = []
                for row in cursor.fetchall():
                    event_data = dict(row)
                    event_data['details'] = json.loads(event_data['details'])
                    event_data['tags'] = json.loads(event_data['tags'] or '[]')
                    events.append(AuditEvent.from_dict(event_data))
                
                return events
                
        except Exception as e:
            logger.error(f"Failed to query audit events: {e}")
            return []
    
    def get_event_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get audit event statistics"""
        start_date = (datetime.now(timezone.utc) - timedelta(days=days)).date()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Overall statistics
                total_events = conn.execute(
                    "SELECT COUNT(*) as count FROM audit_events WHERE date(timestamp) >= ?",
                    (start_date.isoformat(),)
                ).fetchone()['count']
                
                # Events by type
                type_stats = conn.execute('''
                    SELECT event_type, COUNT(*) as count
                    FROM audit_events
                    WHERE date(timestamp) >= ?
                    GROUP BY event_type
                    ORDER BY count DESC
                ''', (start_date.isoformat(),)).fetchall()
                
                # Events by severity
                severity_stats = conn.execute('''
                    SELECT severity, COUNT(*) as count
                    FROM audit_events
                    WHERE date(timestamp) >= ?
                    GROUP BY severity
                    ORDER BY count DESC
                ''', (start_date.isoformat(),)).fetchall()
                
                # Events by outcome
                outcome_stats = conn.execute('''
                    SELECT outcome, COUNT(*) as count
                    FROM audit_events
                    WHERE date(timestamp) >= ?
                    GROUP BY outcome
                    ORDER BY count DESC
                ''', (start_date.isoformat(),)).fetchall()
                
                # Top users by events
                user_stats = conn.execute('''
                    SELECT user_id, COUNT(*) as count
                    FROM audit_events
                    WHERE date(timestamp) >= ?
                    GROUP BY user_id
                    ORDER BY count DESC
                    LIMIT 10
                ''', (start_date.isoformat(),)).fetchall()
                
                # High-risk events
                high_risk_events = conn.execute('''
                    SELECT COUNT(*) as count
                    FROM audit_events
                    WHERE date(timestamp) >= ? AND risk_score >= 70
                ''', (start_date.isoformat(),)).fetchone()['count']
                
                return {
                    "period_days": days,
                    "total_events": total_events,
                    "events_by_type": [dict(row) for row in type_stats],
                    "events_by_severity": [dict(row) for row in severity_stats],
                    "events_by_outcome": [dict(row) for row in outcome_stats],
                    "top_users": [dict(row) for row in user_stats],
                    "high_risk_events": high_risk_events,
                    "risk_percentage": (high_risk_events / total_events * 100) if total_events > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get audit statistics: {e}")
            return {}
    
    def export_events(self, filepath: str, start_time: datetime = None,
                     end_time: datetime = None, format_type: str = "json") -> None:
        """Export audit events to file"""
        events = self.query_events(start_time=start_time, end_time=end_time, limit=100000)
        
        export_data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "total_events": len(events),
            "events": [event.to_dict() for event in events]
        }
        
        try:
            if format_type.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            logger.info(f"Audit events exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export audit events: {e}")
            raise

# Global audit logger
_audit_logger = None

def get_audit_logger() -> AuditLogger:
    """Get the global audit logger"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger

def log_security_event(event_type: EventType, severity: EventSeverity, user_id: str,
                      resource: str, action: str, outcome: str, **kwargs) -> str:
    """Log security event - convenience function"""
    return get_audit_logger().log_event(event_type, severity, user_id, resource, action, outcome, **kwargs)

def log_access_attempt(user_id: str, resource: str, action: str, success: bool, **kwargs) -> str:
    """Log access attempt"""
    outcome = "success" if success else "failure"
    severity = EventSeverity.MEDIUM if not success else EventSeverity.LOW
    return log_security_event(
        EventType.AUTHORIZATION, severity, user_id, resource, action, outcome, **kwargs
    )

def log_data_access(user_id: str, resource: str, data_type: str, sensitive: bool = False, **kwargs) -> str:
    """Log data access"""
    severity = EventSeverity.HIGH if sensitive else EventSeverity.LOW
    details = kwargs.get('details', {})
    details['data_type'] = data_type
    details['sensitive_data_accessed'] = sensitive
    kwargs['details'] = details
    
    return log_security_event(
        EventType.DATA_ACCESS, severity, user_id, resource, "read", "success", **kwargs
    )