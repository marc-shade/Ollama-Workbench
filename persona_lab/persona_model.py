from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional
import uuid
import json
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('persona_lab.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('persona_lab')

@dataclass
class PersonaHistory:
    persona_id: str
    field_name: str
    old_value: str
    new_value: str
    timestamp: datetime
    modified_by: str

@dataclass
class Persona:
    id: str
    name: str
    age: int
    nationality: str
    occupation: str
    background: str
    routine: str
    personality: str
    skills: List[str]
    avatar: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 150
    created_at: datetime = None
    modified_at: datetime = None
    version: int = 1
    tags: List[str] = None
    notes: str = ""
    generated_by: str = "AI"  # AI or Manual
    interaction_history: List[Dict] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.modified_at is None:
            self.modified_at = self.created_at
        if self.tags is None:
            self.tags = []
        if self.interaction_history is None:
            self.interaction_history = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self):
        return {
            **asdict(self),
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat(),
            'skills': json.dumps(self.skills),
            'tags': json.dumps(self.tags),
            'interaction_history': json.dumps(self.interaction_history),
            'metadata': json.dumps(self.metadata)
        }

    @classmethod
    def from_dict(cls, data):
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['modified_at'] = datetime.fromisoformat(data['modified_at'])
        data['skills'] = json.loads(data['skills'])
        data['tags'] = json.loads(data['tags'])
        data['interaction_history'] = json.loads(data['interaction_history'])
        data['metadata'] = json.loads(data['metadata'])
        return cls(**data)

class PersonaDB:
    def __init__(self, db_path: str = "personas/persona_lab.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(exist_ok=True)
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS personas (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    age INTEGER,
                    nationality TEXT,
                    occupation TEXT,
                    background TEXT,
                    routine TEXT,
                    personality TEXT,
                    skills TEXT,
                    avatar TEXT,
                    model TEXT,
                    temperature REAL,
                    max_tokens INTEGER,
                    created_at TEXT,
                    modified_at TEXT,
                    version INTEGER,
                    tags TEXT,
                    notes TEXT,
                    generated_by TEXT,
                    interaction_history TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS persona_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    persona_id TEXT,
                    field_name TEXT,
                    old_value TEXT,
                    new_value TEXT,
                    timestamp TEXT,
                    modified_by TEXT,
                    FOREIGN KEY (persona_id) REFERENCES personas (id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS persona_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE
                )
            """)

    def create_persona(self, persona: Persona) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO personas
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    tuple(persona.to_dict().values())
                )
                logger.info(f"Created new persona: {persona.name} (ID: {persona.id})")
                return True
        except Exception as e:
            logger.error(f"Error creating persona: {str(e)}")
            return False

    def update_persona(self, persona: Persona, modified_by: str = "system") -> bool:
        try:
            # Get the old persona data
            old_persona = self.get_persona(persona.id)
            if old_persona is None:
                return False

            # Record changes in history
            with sqlite3.connect(self.db_path) as conn:
                old_dict = old_persona.to_dict()
                new_dict = persona.to_dict()
                for key in old_dict:
                    if old_dict[key] != new_dict[key]:
                        conn.execute(
                            """
                            INSERT INTO persona_history
                            (persona_id, field_name, old_value, new_value, timestamp, modified_by)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (persona.id, key, str(old_dict[key]), str(new_dict[key]),
                             datetime.now().isoformat(), modified_by)
                        )

                # Update the persona
                persona.version += 1
                persona.modified_at = datetime.now()
                conn.execute(
                    """
                    UPDATE personas
                    SET name=?, age=?, nationality=?, occupation=?, background=?,
                        routine=?, personality=?, skills=?, avatar=?, model=?,
                        temperature=?, max_tokens=?, modified_at=?, version=?,
                        tags=?, notes=?, generated_by=?, interaction_history=?,
                        metadata=?
                    WHERE id=?
                    """,
                    (*tuple(persona.to_dict().values())[1:-1], persona.id)
                )
                logger.info(f"Updated persona: {persona.name} (ID: {persona.id})")
                return True
        except Exception as e:
            logger.error(f"Error updating persona: {str(e)}")
            return False

    def delete_persona(self, persona_id: str) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM personas WHERE id=?", (persona_id,))
                conn.execute("DELETE FROM persona_history WHERE persona_id=?", (persona_id,))
                logger.info(f"Deleted persona with ID: {persona_id}")
                return True
        except Exception as e:
            logger.error(f"Error deleting persona: {str(e)}")
            return False

    def get_persona(self, persona_id: str) -> Optional[Persona]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM personas WHERE id=?", (persona_id,))
                row = cursor.fetchone()
                if row:
                    return Persona.from_dict(dict(zip([col[0] for col in cursor.description], row)))
                return None
        except Exception as e:
            logger.error(f"Error getting persona: {str(e)}")
            return None

    def get_all_personas(self) -> List[Persona]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM personas")
                return [
                    Persona.from_dict(dict(zip([col[0] for col in cursor.description], row)))
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Error getting all personas: {str(e)}")
            return []

    def get_persona_history(self, persona_id: str) -> List[PersonaHistory]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM persona_history WHERE persona_id=? ORDER BY timestamp DESC",
                    (persona_id,)
                )
                return [
                    PersonaHistory(
                        persona_id=row[1],
                        field_name=row[2],
                        old_value=row[3],
                        new_value=row[4],
                        timestamp=datetime.fromisoformat(row[5]),
                        modified_by=row[6]
                    )
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Error getting persona history: {str(e)}")
            return []

    def add_tag(self, tag_name: str) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT OR IGNORE INTO persona_tags (name) VALUES (?)", (tag_name,))
                return True
        except Exception as e:
            logger.error(f"Error adding tag: {str(e)}")
            return False

    def get_all_tags(self) -> List[str]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT name FROM persona_tags")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting tags: {str(e)}")
            return []

    def search_personas(self, query: str) -> List[Persona]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT * FROM personas
                    WHERE name LIKE ? OR occupation LIKE ? OR nationality LIKE ?
                    OR background LIKE ? OR personality LIKE ? OR notes LIKE ?
                    """,
                    tuple(['%' + query + '%'] * 6)
                )
                return [
                    Persona.from_dict(dict(zip([col[0] for col in cursor.description], row)))
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Error searching personas: {str(e)}")
            return []
