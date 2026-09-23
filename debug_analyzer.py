import sys
import sqlite3
import json
import os
import traceback

print("DEBUG: script started", flush=True)

try:
    from groq import Groq
    print("DEBUG: groq imported OK", flush=True)
except Exception as e:
    print(f"DEBUG: FAILED to import groq -> {e}", flush=True)
    sys.exit(1)

DB_PATH = "disha.db"

api_key = os.environ.get("GROQ_API_KEY")
print(f"DEBUG: GROQ_API_KEY present: {bool(api_key)}, length: {len(api_key) if api_key else 0}", flush=True)

try:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    print(f"DEBUG: connected to {DB_PATH}", flush=True)

    cur.execute("SELECT COUNT(*) FROM candidates")
    count = cur.fetchone()[0]
    print(f"DEBUG: candidates table has {count} rows", flush=True)

    cur.execute("SELECT id, resume_text FROM candidates LIMIT 2")
    rows = cur.fetchall()
    print(f"DEBUG: fetched {len(rows)} rows for testing", flush=True)

    if not rows:
        print("DEBUG: No rows returned — stopping here.", flush=True)
        sys.exit(0)

    print("DEBUG: creating Groq client...", flush=True)
    client = Groq(api_key=api_key)
    print("DEBUG: Groq client created OK", flush=True)

    candidate_id, resume_text = rows[0]
    print(f"DEBUG: testing on candidate {candidate_id}, resume length {len(resume_text)}", flush=True)

    print("DEBUG: calling Groq API...", flush=True)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You output only valid JSON: {\"test\": \"ok\"}"},
            {"role": "user", "content": "respond now"},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    print("DEBUG: API call returned!", flush=True)
    print("DEBUG: raw response:", response.choices[0].message.content, flush=True)

except Exception as e:
    print("DEBUG: EXCEPTION CAUGHT:", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("DEBUG: script finished successfully", flush=True)