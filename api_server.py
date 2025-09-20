from fastapi import FastAPI, Request
from pydantic import BaseModel
import sqlite3

app = FastAPI()

# Database setup
conn = sqlite3.connect("incidents.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emp_id TEXT,
    date_time TEXT
)
""")
conn.commit()

class Incident(BaseModel):
    emp_id: str
    date_time: str

@app.post("/incident")
async def save_incident(incident: Incident):
    cursor.execute(
        "INSERT INTO incidents (emp_id, date_time) VALUES (?, ?)",
        (incident.emp_id, incident.date_time)
    )
    conn.commit()
    return {"status": "success", "emp_id": incident.emp_id, "date_time": incident.date_time}