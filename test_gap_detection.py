"""
Manual gap-detection test — run this once to confirm the Analyzer Agent
correctly extracts gap_duration and gap_context, since none of the real
seed resumes happen to contain one.
"""
import os
import json
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a resume skill-extraction engine. You output ONLY valid JSON, nothing else — no preamble, no markdown code fences, no explanation.

Given a resume's text, extract:
1. "skills": a flat list of concrete skills (technical AND soft skills), inferred from experience described, not just explicitly labeled "Skills" sections.
2. "years_experience": your best estimate of total years of relevant work experience, as a number.
3. "gap_duration": if the resume text mentions an employment gap, state its approximate length (e.g. "2 years", "8 months"). If no gap is mentioned, use null.
4. "gap_context": a short (under 15 words) neutral, factual description of the reason for the gap if stated. If not mentioned, use null. Never speculate or invent a reason.

Output this exact JSON schema and nothing else:
{"skills": ["skill1", "skill2"], "years_experience": <number>, "gap_duration": "<string or null>", "gap_context": "<string or null>"}
"""

TEST_RESUMES = [
    "Senior Software Engineer with 6 years experience in Java and Spring Boot. Took a 3-year career break to care for aging parents. Now seeking to return to a backend engineering role.",
    "Marketing Manager, 8 years at various agencies. Left the workforce for 18 months following the birth of my second child. Managed brand campaigns, budgets up to $2M, and a team of 6.",
    "Accountant with CPA certification, 10 years experience in corporate finance. Relocated internationally with spouse's job transfer, resulting in a 14-month employment gap. Proficient in SAP and QuickBooks.",
]

for i, resume in enumerate(TEST_RESUMES, 1):
    print(f"\n--- Gap Test {i} ---")
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Resume text: {resume}"},
        ],
        temperature=0.1,
        max_tokens=1024,
        reasoning_effort="low",
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    print(json.dumps(result, indent=2))