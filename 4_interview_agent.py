"""
INTERVIEWER AGENT + RED FLAG RADAR 
Runs a text-based mock interview (voice/Whisper wiring comes after this logic
is proven — same pattern as the Analyzer Agent). Every question the recruiter
persona asks gets classified {neutral | borderline | discriminatory}; flagged
ones get a plain-language legal-literacy note.
"""

import sqlite3
import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
DB_PATH = "disha.db"
MODEL = "openai/gpt-oss-20b"

RECRUITER_SYSTEM_PROMPT_WITH_GAP = """You are playing the role of a recruiter conducting a mock interview with a candidate who has a real career gap on their resume. Your job is to ask ONE realistic interview question at a time, adapting to the candidate's previous answer.

Candidate context:
{candidate_context}

Rules:
- Ask exactly ONE question per turn. No preamble, no "Great, thanks!" — just the question itself.
- Reference specifics from the candidate's actual skills/experience where relevant.
- At some point in the interview (not necessarily first), ask a question that probes the gap in a way a real, imperfect recruiter sometimes does — specifically framing it as a risk tied to the personal reason for the gap. For example, if the gap was for childcare, ask something like "how will you manage this role now that you have a young child at home?" or "are you planning on having more children soon?" — a question that ties the candidate's family/personal circumstances to their perceived reliability or commitment. This is intentional: it needs to be concrete enough for a legal-literacy classifier to correctly flag it, not just a neutral "tell me about your gap" question.
- Keep questions concise, one to two sentences.
- Never break character or mention that this is a simulation.
"""

RECRUITER_SYSTEM_PROMPT_NO_GAP = """You are playing the role of a recruiter conducting a mock interview with a candidate. Your job is to ask ONE realistic interview question at a time, adapting to the candidate's previous answer.

Candidate context:
{candidate_context}

Rules:
- Ask exactly ONE question per turn. No preamble, no "Great, thanks!" — just the question itself.
- Reference specifics from the candidate's actual skills/experience where relevant.
- This candidate has no employment gap — do NOT invent or imply one. Focus entirely on their actual experience, skills, and career progression.
- Keep questions concise, one to two sentences.
- Never break character or mention that this is a simulation.
"""

CLASSIFIER_SYSTEM_PROMPT = """You are an employment-law-aware classifier. Given a single interview question asked of a candidate with a resume gap, classify it as one of:
- "neutral": a normal, appropriate interview question
- "borderline": touches on the gap or personal circumstances but isn't clearly illegal, just worth a coaching note
- "discriminatory": asks about protected/personal circumstances in a way that could be illegal or inappropriate (e.g. marriage/family plans, childcare arrangements framed as a job risk, direct implication that the gap reflects poorly on commitment)

Output ONLY valid JSON: {"flag_type": "neutral" | "borderline" | "discriminatory", "citation": "<one-sentence plain-language note on the relevant right/protection, or null if neutral>"}
"""


def get_candidate_context(conn, candidate_id):
    cur = conn.cursor()
    cur.execute("SELECT category, gap_duration, gap_context FROM candidates WHERE id = ?", (candidate_id,))
    category, gap_duration, gap_context = cur.fetchone()
    cur.execute("SELECT skill FROM skills_extracted WHERE candidate_id = ?", (candidate_id,))
    skills = [r[0] for r in cur.fetchall()]
    context = (
        f"Background: {category}. Skills: {', '.join(skills[:10])}. "
        f"Career gap: {gap_duration or 'none stated'}, reason: {gap_context or 'not stated'}."
    )
    has_gap = bool(gap_duration)
    return context, has_gap


def ask_question(client, candidate_context, conversation_history, has_gap, max_retries=2):
    prompt_template = RECRUITER_SYSTEM_PROMPT_WITH_GAP if has_gap else RECRUITER_SYSTEM_PROMPT_NO_GAP
    messages = [
        {"role": "system", "content": prompt_template.format(candidate_context=candidate_context)}
    ] + conversation_history

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.8,
                max_tokens=1024,
                reasoning_effort="medium",  # dialogue needs more room than one-shot extraction did
            )
            content = response.choices[0].message.content.strip()
            if content:  # guard against empty-but-successful responses too
                return content
        except Exception as e:
            print(f"   (retrying question generation, attempt {attempt + 1} failed: {type(e).__name__})")

    # Final fallback so the interview never hard-crashes on stage
    return "Tell me more about a time you handled a challenging situation in your last role."


def classify_question(client, question, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {question}"},
                ],
                temperature=0.1,
                max_tokens=1024,
                reasoning_effort="low",
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"   (retrying classification, attempt {attempt + 1} failed: {type(e).__name__})")

    return {"flag_type": "neutral", "citation": None}  # fail safe, doesn't block the interview


def run_interview(candidate_id: int, num_turns: int = 5):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    candidate_context, has_gap = get_candidate_context(conn, candidate_id)
    print(f"\n=== Mock Interview — Candidate {candidate_id} ===")
    print(f"Context: {candidate_context}")
    print(f"(Gap-probing questions enabled: {has_gap})\n")

    conversation_history = []

    for turn in range(1, num_turns + 1):
        question = ask_question(client, candidate_context, conversation_history, has_gap)
        print(f"\n🎙️  Recruiter (Q{turn}): {question}")

        classification = classify_question(client, question)
        flag_type = classification.get("flag_type", "neutral")
        citation = classification.get("citation")

        if flag_type != "neutral":
            print(f"   ⚠️  [{flag_type.upper()}] {citation}")

        answer = input("💬 Your answer: ")

        cur.execute(
            "INSERT INTO interview_transcripts (candidate_id, question, answer, turn_number) VALUES (?, ?, ?, ?)",
            (candidate_id, question, answer, turn),
        )
        conn.commit()
        transcript_id = cur.lastrowid

        if flag_type != "neutral":
            cur.execute(
                "INSERT INTO flagged_questions (transcript_id, flag_type, citation) VALUES (?, ?, ?)",
                (transcript_id, flag_type, citation),
            )
            conn.commit()

        conversation_history.append({"role": "assistant", "content": question})
        conversation_history.append({"role": "user", "content": answer})

    conn.close()
    print(f"\n✅ Interview complete. {num_turns} turns saved to interview_transcripts.")


if __name__ == "__main__":
    run_interview(candidate_id=51, num_turns=5)