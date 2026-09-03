import sqlite3
import os
import json
import logging

logger = logging.getLogger(__name__)

DB_PATH = "leads.db"

def get_db_connection():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        import psycopg2
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return psycopg2.connect(db_url)
    else:
        sqlite_file = os.getenv("SQLITE_DB_PATH", DB_PATH)
        conn = sqlite3.connect(sqlite_file)
        conn.row_factory = sqlite3.Row
        return conn

def get_cursor(conn):
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        from psycopg2.extras import RealDictCursor
        return conn.cursor(cursor_factory=RealDictCursor)
    else:
        return conn.cursor()

def init_db():
    """Initializes the database schema."""
    conn = get_db_connection()
    cursor = get_cursor(conn)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            phone TEXT,
            company TEXT,
            country TEXT,
            interests TEXT,
            message TEXT,
            website TEXT,
            linkedin_url TEXT,
            archetype TEXT,
            industry TEXT,
            executive_summary TEXT,
            raw_dossier TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def insert_lead(dossier: dict):
    """Inserts a researched lead dossier into the database."""
    conn = get_db_connection()
    cursor = get_cursor(conn)
    try:
        name = dossier.get("name") or dossier.get("lead_name") or "Executive Lead"
        email = dossier.get("email") or dossier.get("lead_email") or ""
        phone = dossier.get("phone") or ""
        company = dossier.get("company") or dossier.get("company_name") or "Target Enterprise"
        country = dossier.get("country") or ""
        interests = dossier.get("interests") or dossier.get("referred_product") or ""
        message = dossier.get("message") or dossier.get("use_case") or ""
        website = dossier.get("website") or ""
        linkedin = dossier.get("linkedin_url") or ""
        archetype = dossier.get("archetype") or ""
        industry = dossier.get("industry") or dossier.get("industry_focus") or ""
        exec_sum = dossier.get("professional_summary") or dossier.get("executive_profile_analysis") or ""
        raw_json = json.dumps(dossier, ensure_ascii=False)

        cursor.execute("""
            INSERT INTO leads (name, email, phone, company, country, interests, message, website, linkedin_url, archetype, industry, executive_summary, raw_dossier)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, email, phone, company, country, interests, message, website, linkedin, archetype, industry, exec_sum, raw_json))
        conn.commit()
    except Exception as e:
        logger.error(f"Error inserting lead: {e}")
    finally:
        conn.close()

def get_all_leads():
    """Returns all stored leads as a list of dicts."""
    conn = get_db_connection()
    cursor = get_cursor(conn)
    try:
        cursor.execute("SELECT id, name, company, email, phone, country, archetype, industry, created_at FROM leads ORDER BY id DESC")
        rows = cursor.fetchall()
        leads = [dict(r) for r in rows]
        return leads
    except Exception as e:
        logger.error(f"Error fetching leads: {e}")
        return []
    finally:
        conn.close()

def get_lead_by_id(lead_id: int):
    """Fetches a lead dossier by ID."""
    conn = get_db_connection()
    cursor = get_cursor(conn)
    try:
        cursor.execute("SELECT raw_dossier FROM leads WHERE id = ?", (lead_id,))
        row = cursor.fetchone()
        if row and row["raw_dossier"]:
            return json.loads(row["raw_dossier"])
        return None
    except Exception as e:
        logger.error(f"Error fetching lead {lead_id}: {e}")
        return None
    finally:
        conn.close()

def delete_lead(lead_id: int):
    """Deletes a lead by ID."""
    conn = get_db_connection()
    cursor = get_cursor(conn)
    try:
        cursor.execute("DELETE FROM leads WHERE id = ?", (lead_id,))
        conn.commit()
    except Exception as e:
        logger.error(f"Error deleting lead {lead_id}: {e}")
    finally:
        conn.close()
