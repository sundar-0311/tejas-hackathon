"""
RESUME ANALYSER AGENT

Takes a candidate's resume text and returns structured skill data.
Run against all seed candidates in disha.db, writes results to skills_extracted.

"""
import sqlite3
import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # reads GROQ_API_KEY from .env into the environment
DB_PATH = "disha.db"

SYSTEM_PROMPT = """You are a resume skill-extraction engine. You output ONLY valid JSON, nothing else — no preamble, no markdown code fences, no explanation.

Given a resume's text, extract:
1. "skills": a flat list of concrete skills (technical AND soft skills), inferred from experience described, not just explicitly labeled "Skills" sections. Normalize skill names (e.g. "MS Excel" -> "Excel", "Pandas/Numpy" -> two separate entries: "Pandas", "NumPy").
2. "years_experience": your best estimate of total years of relevant work experience, as a number. If unclear, estimate conservatively from dates mentioned.
3. "gap_duration": if the resume text or accompanying context mentions an employment gap, state its approximate length (e.g. "2 years", "8 months"). If no gap is mentioned, use null.
4. "gap_context": a short (under 15 words) neutral, factual description of the reason for the gap if stated (e.g. "childcare", "eldercare", "relocation"). If not mentioned, use null. Never speculate or invent a reason.

Output this exact JSON schema and nothing else:
{
  "skills": ["skill1", "skill2", ...],
  "years_experience": <number>,
  "gap_duration": "<string or null>",
  "gap_context": "<string or null>"
}

Rules:
- Never wrap the JSON in markdown fences.
- Never add commentary before or after the JSON.
- If you are unsure about a field, use your best reasonable estimate rather than refusing — but never fabricate a gap or reason that isn't supported by the text.
"""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "Resume text: 'Marketing coordinator, 3 years at ABC Corp, managed social media campaigns, Excel reporting, basic SQL for analytics. Took a 2-year career break for childcare, now seeking to return.'"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "skills": ["Social Media Marketing", "Excel", "SQL", "Campaign Management", "Analytics Reporting"],
            "years_experience": 3,
            "gap_duration": "2 years",
            "gap_context": "childcare"
        })
    },
    {
        "role": "user",
        "content": "Resume text: 'Software Engineer with 5 years experience in Python, Django, REST APIs, PostgreSQL. Led a team of 3 junior developers. No employment gaps.'"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "skills": ["Python", "Django", "REST APIs", "PostgreSQL", "Team Leadership"],
            "years_experience": 5,
            "gap_duration": None,
            "gap_context": None
        })
    },
]


def extract_skills(client: Groq, resume_text: str) -> dict:
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + FEW_SHOT_EXAMPLES
        + [{"role": "user", "content": f"Resume text: {resume_text[:4000]}"}]  # truncate very long resumes
    )
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0.1,
        max_tokens= 2048,
        reasoning_effort= "low",
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"  ⚠️  Failed to parse JSON, got: {raw[:200]}")
        return {"skills": [], "years_experience": None, "gap_duration": None, "gap_context": None}


def run_on_seed_candidates(limit: int = 10):
    """Test against a handful of seeded resumes first — per the roadmap, prove the
    prompt works on 10 diverse resumes before wiring it into the full pipeline."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT id, resume_text FROM candidates LIMIT ?", (limit,))
    rows = cur.fetchall()

    for candidate_id, resume_text in rows:
        print(f"\n--- Candidate {candidate_id} ---")
        result = extract_skills(client, resume_text)
        print(json.dumps(result, indent=2))

        # Update candidate's gap fields
        cur.execute(
            "UPDATE candidates SET gap_duration = ?, gap_context = ? WHERE id = ?",
            (result.get("gap_duration"), result.get("gap_context"), candidate_id),
        )

        # Insert each extracted skill as its own row
        for skill in result.get("skills", []):
            cur.execute(
                "INSERT INTO skills_extracted (candidate_id, skill, confidence, source) VALUES (?, ?, ?, ?)",
                (candidate_id, skill, 0.8, "resume"),
            )

    conn.commit()
    conn.close()
    print(f"\n✅ Processed {len(rows)} candidates. Check the skills_extracted table.")


if __name__ == "__main__":
    run_on_seed_candidates(limit=10)