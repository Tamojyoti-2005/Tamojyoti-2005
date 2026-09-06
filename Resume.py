import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_text_from_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in document.paragraphs]
    return "\n".join(paragraphs).strip()


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore").strip()


def extract_resume_text(uploaded_file) -> str:
    """Dispatch to the right extractor based on file extension."""
    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if name.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    elif name.endswith(".docx"):
        text = extract_text_from_docx(file_bytes)
    elif name.endswith(".txt"):
        text = extract_text_from_txt(file_bytes)
    else:
        raise ValueError("Unsupported file type. Upload a PDF, DOCX, or TXT file.")

    if not text:
        raise ValueError("Could not find readable text in that file. Try pasting the text instead.")

    return text


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an experienced technical recruiter and hiring manager. You will be given a \
candidate's resume text and a target job role. Respond with ONLY valid JSON, no markdown code fences, \
no commentary before or after. Use exactly this schema:

{
  "matchPercentage": <integer 0-100, overall fit of this resume for this role>,
  "summary": "<2-3 sentence plain-language summary of the fit, written to the candidate>",
  "matchedSkills": ["<skill or qualification present in the resume that is relevant to the role>", ...],
  "missingSkills": ["<important skill, tool, or qualification for this role that is NOT evident in the resume>", ...],
  "interviewQuestions": [
    {"category": "<short label like 'Technical', 'Behavioral', 'System Design', 'Role-specific'>", "question": "<the interview question>"},
    ...
  ]
}

Rules:
- matchedSkills and missingSkills should each have between 4 and 10 short items (skill/tool/technology names, not full sentences).
- Base missingSkills on what is genuinely important for the given role, even if the resume doesn't mention it.
- Include 7-8 interviewQuestions, mixing technical/role-specific questions with 1-2 behavioral ones. Where possible, tailor a couple of the questions to gaps you identified or to specific experience on the resume.
- Be honest and specific rather than generic. Do not pad lists with vague filler.
- Return nothing but the JSON object."""


@dataclass
class AnalysisResult:
    match_percentage: int
    summary: str
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    interview_questions: List[dict] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: dict) -> "AnalysisResult":
        return cls(
            match_percentage=max(0, min(100, int(data.get("matchPercentage", 0)))),
            summary=data.get("summary", ""),
            matched_skills=data.get("matchedSkills", []) or [],
            missing_skills=data.get("missingSkills", []) or [],
            interview_questions=data.get("interviewQuestions", []) or [],
        )


def extract_json_object(raw: str) -> dict:
    """Extract the top-level JSON analysis object from model output that may include
    reasoning text or markdown fences.

    Scans left-to-right (not right-to-left) so it finds the OUTER object first —
    scanning from the end grabs the last small nested object instead (e.g. the final
    entry in interviewQuestions), which silently returns an empty/near-empty result.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("No response text received from the model.")

    text = text.replace("```json", "").replace("```", "").strip()
    decoder = json.JSONDecoder()

    candidates = []
    for index, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            candidates.append(obj)

    if not candidates:
        raise ValueError("Model did not return valid JSON content.")

    # Prefer an object that actually matches our schema over any stray dict
    # (e.g. a nested question object) the model's reasoning trace might contain.
    for obj in candidates:
        if "matchPercentage" in obj:
            return obj

    return candidates[0]


def analyze_resume(resume_text: str, role: str, api_key: str) -> AnalysisResult:
    """Call the model and parse its JSON analysis.

    Note: reasoning/thinking mode is deliberately OFF here. With thinking on,
    the model spends part of its token budget on a hidden reasoning trace
    before writing the JSON answer — for a short max_tokens budget that trace
    can eat the whole budget and cut the JSON off mid-object, which is what
    produced "Model did not return valid JSON content." This task doesn't
    need visible reasoning, just the structured output, so it's turned off
    and the token budget is raised as a safety margin. If the response still
    comes back truncated (finish_reason == "length"), we retry once with a
    larger budget before giving up.
    """
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)

    def call(max_tokens: int, thinking: bool):
        return client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Target role: {role}\n\nResume text:\n{resume_text[:12000]}",
                },
            ],
            temperature=0.4,
            top_p=0.95,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"thinking": thinking, "reasoning_effort": "low"}},
            stream=False,
        )

    attempts = [(3072, False), (6144, False)]
    last_error = None

    for max_tokens, thinking in attempts:
        completion = call(max_tokens, thinking)
        choice = completion.choices[0]
        content = (choice.message.content or "").strip()
        # Some reasoning models return the trace in a separate field even
        # with thinking off; fall back to it only if content is empty.
        reasoning = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None)

        if not content and reasoning:
            content = reasoning.strip()

        if not content:
            last_error = "The model returned an empty response."
            continue

        try:
            data = extract_json_object(content)
            return AnalysisResult.from_json(data)
        except ValueError as exc:
            last_error = str(exc)
            if choice.finish_reason == "length":
                last_error += " (response was cut off before finishing — retrying with more room.)"
                continue
            # Not a truncation issue, so a bigger budget won't help; stop here.
            break

    snippet = (content[:300] + "…") if content and len(content) > 300 else content
    raise ValueError(f"{last_error} Raw response started with: {snippet!r}" if snippet else last_error)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Resume Check", page_icon="🎯", layout="centered")

PALETTE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --paper: #F5F1E7;
    --ink: #1C1A16;
    --ink-soft: #6B6558;
    --line: rgba(28,26,22,0.12);
    --teal: #0E8F6F;
    --teal-soft: #DEF3EB;
    --rust: #C1432A;
    --rust-soft: #FBE6E0;
    --amber: #B8791A;
    --navy: #1B2A41;
    --navy-2: #0F3D3E;
    --lime: #8FE388;
    --navy-soft: #E4EAF1;
}

html, body, .stApp { background-color: var(--paper) !important; }
* { font-family: 'Inter', sans-serif; }

/* ---- Hero banner ---- */
.hero {
    background: linear-gradient(120deg, var(--navy) 0%, var(--navy-2) 100%);
    border-radius: 14px;
    padding: 40px 36px 34px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(15,25,40,0.25);
}
.hero::after {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(143,227,136,0.28) 0%, rgba(143,227,136,0) 70%);
}
.hero-kicker {
    font-size: 13px;
    font-weight: 600;
    color: #DDF7E6;
    letter-spacing: 0.02em;
    margin-bottom: 10px;
}
.hero-title {
    font-family: 'Source Serif 4', serif;
    font-weight: 700;
    font-size: 36px;
    color: #FFFFFF;
    margin: 0 0 10px;
    line-height: 1.15;
}
.hero-sub {
    font-size: 15px;
    color: #F5F7FA;
    max-width: 52ch;
    line-height: 1.6;
    margin: 0;
}

/* ---- Intake / general card ---- */
.stApp .block-container { padding-top: 2.2rem; }

.stApp .stMarkdown p,
.stApp .stMarkdown div,
.stApp .stTextInput label,
.stApp .stTextArea label,
.stApp .stFileUploader label,
.stApp .stForm,
.stApp .stForm p,
.stApp .stButton p,
.stApp .stCheckbox label,
.stApp .stRadio label,
.stApp .stSelectbox label,
.stApp .stNumberInput label {
    color: #1C1A16 !important;
}

/* Hero text sits on a dark gradient, so it needs to win over the dark-ink
   rule above. Matching selectors on element + class beats ".stApp .stMarkdown
   div/p" on specificity, regardless of source order. */
.stApp .stMarkdown div.hero-kicker { color: #DDF7E6 !important; }
.stApp .stMarkdown div.hero-title { color: #FFFFFF !important; }
.stApp .stMarkdown p.hero-sub { color: #F5F7FA !important; }

.stApp, .stApp .stMarkdown, .stApp .stMarkdown p,
.stApp label, .stApp .stTextInput label,
.stApp .stTextArea label, .stApp .stFileUploader label,
.stApp .stSelectbox label, .stApp .stNumberInput label,
.stApp .stRadio > label, .stApp .stCheckbox > label,
.stApp .stForm, .stApp .stForm p,
.stApp .stDataFrame, .stApp .stDataFrame * {
    color: var(--ink) !important;
}

.stApp .stTextInput input,
.stApp .stTextArea textarea,
.stApp .stTextInput textarea,
.stApp .stFileUploader div[data-testid="stFileUploaderDropzone"] {
    color: var(--ink) !important;
}

.stApp .stTextInput input::placeholder,
.stApp .stTextArea textarea::placeholder {
    color: var(--ink-soft) !important;
    opacity: 1 !important;
}

div[data-testid="stFileUploader"], div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    border-radius: 8px !important;
    border: 1px solid rgba(28,26,22,0.18) !important;
    background: rgba(255,255,255,0.7) !important;
}

div[data-testid="stFileUploader"] > section,
div[data-testid="stFileUploader"] > div,
div[data-testid="stTextInput"], div[data-testid="stTextArea"] {
    color: var(--ink) !important;
}

div.stButton > button {
    background: linear-gradient(120deg, var(--teal) 0%, var(--navy-2) 100%);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-weight: 700;
    font-size: 15px;
    letter-spacing: 0.01em;
    box-shadow: 0 6px 16px rgba(14,143,111,0.28);
    transition: transform .12s ease, box-shadow .12s ease;
}
div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 22px rgba(14,143,111,0.35);
}

/* ---- Report card (native st.container border) ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid rgba(28,26,22,0.1) !important;
    border-radius: 14px !important;
    box-shadow: 0 10px 26px rgba(28,26,22,0.07);
    animation: rise .45s ease both;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 8px 6px;
}
@keyframes rise {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.score-label {
    font-family: 'Source Serif 4', serif;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 12px;
    display: block;
    color: var(--ink);
}

.score-summary {
    font-size: 17px;
    line-height: 1.7;
    color: var(--ink);
    margin: 0;
    display: block;
}

.chip {
    display: inline-block;
    font-size: 13px;
    font-weight: 600;
    padding: 7px 13px;
    border-radius: 20px;
    margin: 0 8px 8px 0;
    border: 1px solid transparent;
}
.chip.match { background: var(--teal-soft); color: var(--teal); border-color: rgba(14,143,111,0.25); }
.chip.gap { background: var(--rust-soft); color: var(--rust); border-color: rgba(193,67,42,0.22); }

.section-label {
    font-family: 'Source Serif 4', serif;
    font-size: 19px;
    font-weight: 700;
    margin: 4px 0 2px;
    color: var(--ink);
}
.section-sub { font-size: 13px; color: var(--ink-soft); margin-bottom: 16px; }

.q-item {
    border: 1px solid var(--line);
    border-left: 4px solid var(--navy-2);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
    background: var(--paper);
    transition: border-color .15s ease;
}
.q-item.behavioral { border-left-color: var(--amber); }
.q-cat {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: var(--navy-2);
    background: var(--navy-soft);
    padding: 3px 9px;
    border-radius: 20px;
    margin-bottom: 7px;
}
.q-text { font-size: 14.5px; line-height: 1.55; color: var(--ink); margin: 0; }
</style>
"""
st.markdown(PALETTE_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">FIT SCORE · SKILL GAPS · INTERVIEW PREP</div>
        <div class="hero-title">Resume Check</div>
        <p class="hero-sub">Upload a resume, name the role you're aiming for, and get a straight
        read on the fit — what matches, what's missing, and what to prepare for in the interview.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    api_key = None
    try:
        api_key = st.secrets.get("NVIDIA_API_KEY")
    except Exception:
        pass  # no secrets.toml locally — that's fine
    if not api_key:
        api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        api_key = st.text_input(
            "NVIDIA API key",
            type="password",
            help="Set NVIDIA_API_KEY in Streamlit secrets (cloud) or a .env file (local) to skip this field.",
        )

    uploaded_file = st.file_uploader("Resume", type=["pdf", "docx", "txt"])

    with st.expander("Or paste resume text instead"):
        pasted_text = st.text_area("Resume text", height=180, label_visibility="collapsed")

    role = st.text_input("Target role", placeholder="e.g. Frontend Engineer, Data Analyst, Product Manager")

    run = st.button("Check my fit", type="primary", use_container_width=True)

if run:
    resume_text = ""
    error = None

    if uploaded_file is not None:
        try:
            resume_text = extract_resume_text(uploaded_file)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
    elif pasted_text and pasted_text.strip():
        resume_text = pasted_text.strip()
    else:
        error = "Add a resume first — upload a file or paste the text."

    if not error and not role.strip():
        error = "Enter the role you want to apply for."

    if not error and not api_key:
        error = "Add your NVIDIA API key to run the analysis."

    if error:
        st.markdown(
            f'<div style="background:#FBE6E0; color:#7A2A1D; border:1px solid rgba(193,67,42,0.25); border-radius:10px; padding:14px 16px; font-weight:600;">{error}</div>',
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Reading the resume and comparing it against the role..."):
            try:
                result = analyze_resume(resume_text, role.strip(), api_key)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Something went wrong generating the report: {exc}")
                result = None

        if result:
            score = result.match_percentage
            if score >= 75:
                color, label, badge = "var(--teal)", "Strong match", "🟢"
            elif score >= 50:
                color, label, badge = "var(--amber)", "Partial match", "🟡"
            else:
                color, label, badge = "var(--rust)", "Needs work", "🔴"

            circumference = 2 * 3.14159265 * 52
            offset = circumference * (1 - score / 100)

            with st.container(border=True):
                col1, col2 = st.columns([1, 2.2])
                with col1:
                    st.markdown(
                        f"""
                        <svg width="140" height="140" viewBox="0 0 140 140">
                          <defs>
                            <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                              <stop offset="0%" stop-color="{color}" stop-opacity="1" />
                              <stop offset="100%" stop-color="var(--navy-2)" stop-opacity="1" />
                            </linearGradient>
                          </defs>
                          <circle cx="70" cy="70" r="52" fill="none" stroke="var(--line)" stroke-width="11" />
                          <circle cx="70" cy="70" r="52" fill="none" stroke="url(#ringGrad)" stroke-width="11"
                            stroke-linecap="round"
                            stroke-dasharray="{circumference}"
                            stroke-dashoffset="{offset}"
                            transform="rotate(-90 70 70)" />
                          <text x="70" y="66" text-anchor="middle" font-family="Source Serif 4, serif" font-weight="700" font-size="30" fill="var(--ink)">{score}</text>
                          <text x="70" y="86" text-anchor="middle" font-family="Inter, sans-serif" font-size="12" fill="var(--ink-soft)">percent</text>
                        </svg>
                        """,
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        f'<div class="score-label" style="color:{color}">{badge} {label}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="score-summary">{result.summary}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("---")

                mcol, gcol = st.columns(2)
                with mcol:
                    st.markdown('<div class="section-label" style="font-size:15px;">✓ Matched skills</div>', unsafe_allow_html=True)
                    if result.matched_skills:
                        chips = "".join(f'<span class="chip match">{s}</span>' for s in result.matched_skills)
                        st.markdown(chips, unsafe_allow_html=True)
                    else:
                        st.caption("No clear matches found.")
                with gcol:
                    st.markdown('<div class="section-label" style="font-size:15px;">✗ Missing for this role</div>', unsafe_allow_html=True)
                    if result.missing_skills:
                        chips = "".join(f'<span class="chip gap">{s}</span>' for s in result.missing_skills)
                        st.markdown(chips, unsafe_allow_html=True)
                    else:
                        st.caption("Nothing obvious missing — good coverage.")

                st.markdown("---")
                st.markdown('<div class="section-label">Interview prep</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-sub">Questions shaped by this role and this resume.</div>', unsafe_allow_html=True)

                if result.interview_questions:
                    for q in result.interview_questions:
                        category = q.get("category", "General")
                        question = q.get("question", "")
                        extra_class = " behavioral" if "behav" in category.lower() else ""
                        st.markdown(
                            f'<div class="q-item{extra_class}"><span class="q-cat">{category}</span>'
                            f'<p class="q-text">{question}</p></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No questions were generated for this run.")
