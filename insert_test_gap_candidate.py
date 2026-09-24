import sqlite3

conn = sqlite3.connect("disha.db")
cur = conn.cursor()

cur.execute(
    "INSERT INTO candidates (name, resume_text, category, gap_duration, gap_context) VALUES (?, ?, ?, ?, ?)",
    (
        "test_gap_candidate",
        "Marketing Manager, 8 years at various agencies. Left the workforce for 18 months following the birth of my second child. Managed brand campaigns, budgets up to $2M, and a team of 6.",
        "PUBLIC-RELATIONS",
        "18 months",
        "birth of second child",
    ),
)
candidate_id = cur.lastrowid

skills = ["Marketing Strategy", "Brand Campaign Management", "Budget Management", "Team Leadership", "Project Management"]
for skill in skills:
    cur.execute(
        "INSERT INTO skills_extracted (candidate_id, skill, confidence, source) VALUES (?, ?, ?, ?)",
        (candidate_id, skill, 0.9, "resume"),
    )

conn.commit()
conn.close()
print(f"✅ Inserted test gap candidate with id: {candidate_id}")
print(f"Run the interview against this candidate: change candidate_id={candidate_id} in 4_interview_agent.py's __main__ block")