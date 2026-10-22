"""
SQLite database storage manager for analysis sessions and statistics.
"""

import sqlite3
import json
import os
from typing import Optional, Dict, List, Any
from pathlib import Path

from core.models import AnalysisResult
from utils.logging_utils import logger


def init_db(db_path: str = "data/analysis.db"):
    """Initialize SQLite database tables."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            filename TEXT,
            sport TEXT,
            processing_mode TEXT,
            player_count INTEGER,
            total_events INTEGER,
            duration TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            summary_json TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Initialized SQLite database at {db_path}")


def save_session(result: AnalysisResult, db_path: str = "data/analysis.db"):
    """Save an AnalysisResult summary to SQLite database."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    summary_data = {
        "session_id": result.session_id,
        "sport": result.sport,
        "processing_mode": result.processing_mode,
        "video_info": result.video_info.__dict__ if result.video_info else {},
        "player_ids": result.get_player_ids(),
        "errors": result.errors,
        "warnings": result.warnings
    }
    
    filename = result.video_info.filename if result.video_info else "unknown"
    duration = f"{result.video_info.duration}s" if result.video_info else "0s"
    
    cursor.execute("""
        INSERT OR REPLACE INTO sessions 
        (session_id, filename, sport, processing_mode, player_count, total_events, duration, summary_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result.session_id,
        filename,
        result.sport,
        result.processing_mode,
        result.get_player_count(),
        result.get_total_events(),
        duration,
        json.dumps(summary_data)
    ))
    
    conn.commit()
    conn.close()
    logger.info(f"Saved session {result.session_id} to database.")


def list_sessions(db_path: str = "data/analysis.db") -> List[Dict[str, Any]]:
    """Fetch list of all stored session summaries."""
    if not os.path.exists(db_path):
        return []
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT session_id, filename, sport, processing_mode, player_count, total_events, duration, created_at 
        FROM sessions ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "session_id": r[0],
            "filename": r[1],
            "sport": r[2],
            "processing_mode": r[3],
            "player_count": r[4],
            "total_events": r[5],
            "duration": r[6],
            "created_at": r[7]
        }
        for r in rows
    ]


def delete_session(session_id: str, db_path: str = "data/analysis.db"):
    """Delete session by session_id."""
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
    logger.info(f"Deleted session {session_id} from database.")
