"""
  - job_postings: postings.csv (+ job_skills.csv merged in as `skills`)
  - courses: coursera.csv
  - candidates: Resume.csv (seeded so the Analyzer Agent has test data on day 1)
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = "disha.db"
SEARCH_ROOT = Path("dataset")


def find_file(filename: str) -> Path | None:
    matches = list(SEARCH_ROOT.rglob(filename))
    if not matches:
        print(f"Could not find {filename} under {SEARCH_ROOT}/ — skipping.")
        return None
    if len(matches) > 1:
        print(f"Multiple copies of {filename} found, using: {matches[0]}")
    return matches[0]


def create_schema(conn: sqlite3.Connection):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        resume_text TEXT,
        category TEXT,
        gap_duration TEXT,
        gap_context TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS skills_extracted (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_id INTEGER REFERENCES candidates(id),
        skill TEXT,
        confidence REAL,
        source TEXT
    );

    CREATE TABLE IF NOT EXISTS job_postings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT,
        title TEXT,
        company_id TEXT,
        location TEXT,
        description TEXT,
        skills TEXT,
        embedding TEXT
    );

    CREATE TABLE IF NOT EXISTS courses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        provider TEXT,
        url TEXT,
        skills TEXT,
        price TEXT
    );

    CREATE TABLE IF NOT EXISTS gap_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_id INTEGER REFERENCES candidates(id),
        missing_skill TEXT,
        leverage_score REAL,
        recommended_course_id INTEGER REFERENCES courses(id)
    );

    CREATE TABLE IF NOT EXISTS interview_transcripts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_id INTEGER REFERENCES candidates(id),
        question TEXT,
        answer TEXT,
        turn_number INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS flagged_questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transcript_id INTEGER REFERENCES interview_transcripts(id),
        flag_type TEXT,
        citation TEXT
    );

    CREATE TABLE IF NOT EXISTS returnship_matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_id INTEGER REFERENCES candidates(id),
        program_name TEXT,
        company TEXT,
        match_reason TEXT
    );
    """)
    conn.commit()


def load_job_postings(conn, limit=5000):
    postings_path = find_file("postings.csv")
    skills_path = find_file("job_skills.csv")
    if postings_path is None:
        return

    postings = pd.read_csv(postings_path, low_memory=False)
    # Keep a manageable size of 5000 records
    postings = postings.head(limit)

    keep_cols = [c for c in ["job_id", "title", "company_id", "location", "description"] if c in postings.columns]
    postings = postings[keep_cols].copy()

    if skills_path is not None:
        skills = pd.read_csv(skills_path, low_memory=False)
        if "job_id" in skills.columns and "skill_abr" in skills.columns:
            skills_grouped = skills.groupby("job_id")["skill_abr"].apply(
                lambda s: ", ".join(s.dropna().astype(str))
            ).reset_index().rename(columns={"skill_abr": "skills"})
            postings = postings.merge(skills_grouped, on="job_id", how="left")

    if "skills" not in postings.columns:
        postings["skills"] = None

    postings["embedding"] = None
    postings.to_sql("job_postings", conn, if_exists="append", index=False,
                     dtype=None, method="multi", chunksize=500)
    print(f"  ✅ Loaded {len(postings)} job postings.")


def load_courses(conn):
    path = find_file("coursera.csv")
    if path is None:
        return
    df = pd.read_csv(path, low_memory=False)
    rename_map = {"course_by": "provider"}
    df = df.rename(columns=rename_map)
    keep_cols = [c for c in ["title", "provider", "url", "skills", "price"] if c in df.columns]
    df = df[keep_cols]
    df.to_sql("courses", conn, if_exists="append", index=False, method="multi", chunksize=500)
    print(f"  ✅ Loaded {len(df)} courses.")


def load_resumes(conn, limit=50):
    path = find_file("Resume.csv")
    if path is None:
        return
    df = pd.read_csv(path, low_memory=False)
    # snehaanbhawal's schema: ID, Resume_str, Resume_html, Category
    text_col = "Resume_str" if "Resume_str" in df.columns else "Resume"
    cat_col = "Category" if "Category" in df.columns else None

    df = df.head(limit)  # seed with a manageable batch for day-1 prompt testing
    out = pd.DataFrame({
        "name": [f"test_candidate_{i}" for i in range(len(df))],
        "resume_text": df[text_col],
        "category": df[cat_col] if cat_col else None,
    })
    out.to_sql("candidates", conn, if_exists="append", index=False, method="multi", chunksize=500)
    print(f"  ✅ Loaded {len(out)} seed candidates from resumes.")


def main():
    conn = sqlite3.connect(DB_PATH)
    print(f"Creating schema in {DB_PATH} ...")
    create_schema(conn)

    print("Loading job postings ...")
    load_job_postings(conn)

    print("Loading courses ...")
    load_courses(conn)

    print("Loading seed resumes (for testing the Analyzer Agent prompt) ...")
    load_resumes(conn)

    conn.close()
    print(f"\nDone. {DB_PATH} is ready.")


if __name__ == "__main__":
    main()