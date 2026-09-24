"""
MARKET AGENT (Day 1, Afternoon block)

For a given candidate:
1. Finds the most relevant job postings for their target role (via title embedding similarity)
2. Scans those postings' descriptions for known skill terms (the vocabulary grows organically
   from everything the Analyzer Agent has ever extracted across all candidates)
3. Flags skills frequently required in the market but missing from the candidate's profile
4. Recommends a specific free course for each top gap
5. Writes results to gap_scores
"""

import sqlite3
import json
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
from langdetect import detect, LangDetectException

DB_PATH = "disha.db"
MODEL_NAME = "all-MiniLM-L6-v2"  
CATEGORY_TO_ROLE = {
    "HR": "Human Resources Manager",
    "ENGINEERING": "Engineer",
    "INFORMATION-TECHNOLOGY": "IT professional",
    "FINANCE": "Finance professional",
    "HEALTHCARE": "Healthcare professional",
    "FITNESS": "Fitness Trainer",
    "PUBLIC-RELATIONS": "Public Relations Specialist",
    "SALES": "Sales Representative",
    "TEACHER": "Teacher",
}


def get_model():
    print("Loading embedding model (first run downloads ~80MB, then cached)...")
    return SentenceTransformer(MODEL_NAME)


def get_candidate(conn, candidate_id):
    cur = conn.cursor()
    cur.execute("SELECT id, category, gap_duration, gap_context FROM candidates WHERE id = ?", (candidate_id,))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"No candidate with id {candidate_id}")
    cur.execute("SELECT skill FROM skills_extracted WHERE candidate_id = ?", (candidate_id,))
    skills = [r[0] for r in cur.fetchall()]
    return {"id": row[0], "category": row[1], "gap_duration": row[2], "gap_context": row[3], "skills": skills}


def get_skill_vocabulary(conn, min_count=1):
    """The growing skill vocabulary: every distinct skill any candidate's Analyzer Agent
    run has ever extracted. This is what we scan job descriptions for."""
    cur = conn.cursor()
    cur.execute("SELECT skill, COUNT(*) as c FROM skills_extracted GROUP BY skill HAVING c >= ?", (min_count,))
    return [r[0] for r in cur.fetchall()]


def find_relevant_postings(conn, model, target_role, top_k=50):
    cur = conn.cursor()
    cur.execute("SELECT id, title, description FROM job_postings WHERE title IS NOT NULL")
    rows = cur.fetchall()
    titles = [r[1] for r in rows]

    print(f"Embedding {len(titles)} job titles to find postings relevant to '{target_role}'...")
    title_embeddings = model.encode(titles, convert_to_tensor=True, show_progress_bar=True)
    role_embedding = model.encode(target_role, convert_to_tensor=True)

    scores = util.cos_sim(role_embedding, title_embeddings)[0]
    top_indices = np.argsort(-scores.cpu().numpy())[:top_k]

    return [{"id": rows[i][0], "title": rows[i][1], "description": rows[i][2] or ""} for i in top_indices]


def count_skill_mentions(postings, vocabulary):
    """Simple case-insensitive substring frequency count of each vocabulary skill
    across the relevant postings' descriptions."""
    counts = {}
    for skill in vocabulary:
        pattern = re.escape(skill)
        count = sum(1 for p in postings if re.search(pattern, p["description"], re.IGNORECASE))
        if count > 0:
            counts[skill] = count
    return counts


def find_gaps(model, candidate_skills, market_skill_counts, similarity_threshold=0.6):
    if not candidate_skills:
        return sorted(market_skill_counts.items(), key=lambda x: -x[1])

    candidate_embeddings = model.encode(candidate_skills, convert_to_tensor=True)
    gaps = []
    for skill, count in market_skill_counts.items():
        skill_embedding = model.encode(skill, convert_to_tensor=True)
        max_sim = util.cos_sim(skill_embedding, candidate_embeddings).max().item()
        if max_sim < similarity_threshold:
            gaps.append((skill, count, max_sim))
    gaps.sort(key=lambda x: -x[1])  # rank by leverage (market demand)
    return gaps


def is_english(text):
    """Real language detection instead of guessing from character sets — catches
    Spanish/French course titles even when they contain no accented characters
    or happen to include a few English loanwords (e.g. 'Microsoft', 'Excel')."""
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def recommend_course(conn, model, skill, min_similarity=0.55):
    cur = conn.cursor()
    cur.execute("SELECT id, title, skills FROM courses WHERE skills IS NOT NULL")
    rows = cur.fetchall()
    if not rows:
        return None

    filtered = [r for r in rows if is_english(r[1])]
    if not filtered:
        filtered = rows  # fallback if filtering removed everything

    course_texts = [f"{r[1]} {r[2]}" for r in filtered]
    course_embeddings = model.encode(course_texts, convert_to_tensor=True)
    skill_embedding = model.encode(skill, convert_to_tensor=True)
    scores = util.cos_sim(skill_embedding, course_embeddings)[0]
    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()

    if best_score < min_similarity:
        return None  # no good match — better to say nothing than recommend noise

    return filtered[best_idx][0], filtered[best_idx][1], best_score


def run_market_agent(candidate_id: int, top_n_gaps: int = 8):
    conn = sqlite3.connect(DB_PATH)
    model = get_model()

    candidate = get_candidate(conn, candidate_id)
    target_role = CATEGORY_TO_ROLE.get(candidate["category"], candidate["category"] or "professional")
    print(f"\nCandidate {candidate_id} | Category: {candidate['category']} | Target role: {target_role}")
    print(f"Candidate skills ({len(candidate['skills'])}): {candidate['skills']}")

    vocabulary = get_skill_vocabulary(conn)
    print(f"\nSkill vocabulary size (from all processed resumes so far): {len(vocabulary)}")

    relevant_postings = find_relevant_postings(conn, model, target_role)
    print(f"Found {len(relevant_postings)} relevant postings for this role.")

    market_counts = count_skill_mentions(relevant_postings, vocabulary)
    print(f"\nTop 10 market-demanded skills for this role:")
    for skill, count in sorted(market_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {skill}: mentioned in {count}/{len(relevant_postings)} postings")

    gaps = find_gaps(model, candidate["skills"], market_counts)
    top_gaps = gaps[:top_n_gaps]

    print(f"\n=== TOP {len(top_gaps)} SKILL GAPS (radar chart data) ===")
    cur = conn.cursor()
    for skill, leverage, similarity in top_gaps:
        course = recommend_course(conn, model, skill)
        if course:
            course_id, course_title, course_sim = course
            print(f"  GAP: {skill} (demand: {leverage} postings, closest existing skill match: {similarity:.2f})")
            print(f"    -> Recommended course: {course_title} (match: {course_sim:.2f})")
        else:
            course_id = None
            print(f"  GAP: {skill} (demand: {leverage} postings, closest existing skill match: {similarity:.2f})")
            print(f"    -> No strong course match found in current dataset for this skill.")

        cur.execute(
            "INSERT INTO gap_scores (candidate_id, missing_skill, leverage_score, recommended_course_id) VALUES (?, ?, ?, ?)",
            (candidate_id, skill, leverage, course_id),
        )

    conn.commit()
    conn.close()
    print(f"\n✅ Gap analysis complete for candidate {candidate_id}. {len(top_gaps)} gaps written to gap_scores.")


if __name__ == "__main__":
    run_market_agent(candidate_id=1)