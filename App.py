import os
import re
import json
import ast
import time
import base64

import random
import socket
import secrets
import platform
import datetime
from typing import List, Tuple, Dict

import streamlit as st
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

# PDF text extraction
from pdfminer.high_level import extract_text as pdf_extract_text

# Optional tag-chips UI
try:
    from streamlit_tags import st_tags
    HAS_ST_TAGS = True
except Exception:
    HAS_ST_TAGS = False

# Optional geo (non-blocking)
try:
    import geocoder
    from geopy.geocoders import Nominatim
    GEO_OK = True
except Exception:
    GEO_OK = False

# Keys / models
# API keys and models
import os, streamlit as st

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # optional
# Read Groq API key from Streamlit secrets or environment variable
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_GPT_MODEL = os.getenv("OPENAI_GPT_MODEL", "gpt-4o-mini")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Initialize OpenAI client if available
USE_OPENAI = False
openai_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        USE_OPENAI = True
    except Exception as e:
        USE_OPENAI = False
        st.warning(f"OpenAI client unavailable ({e}).")

# Initialize Groq client if available
USE_GROQ = False
groq_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        USE_GROQ = True
    except Exception:
        USE_GROQ = False

# Courses module (your dataset)
try:
    from Courses import (
        ds_course, web_course, android_course, ios_course,
        uiux_course, resume_videos, interview_videos
    )
except Exception:
    ds_course = web_course = android_course = ios_course = uiux_course = []
    resume_videos = interview_videos = []


# SQLite DB (local, zero-setup)
import sqlite3

@st.cache_resource(show_spinner=False)
def get_db_connection():
    try:
        conn = sqlite3.connect("resume_data.db", check_same_thread=False)
        return conn
    except Exception as e:
        st.error(f"SQLite connection failed: {e}")
        return None

connection = get_db_connection()
if connection:
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_data (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            sec_token TEXT NOT NULL,
            ip_add TEXT,
            host_name TEXT,
            dev_user TEXT,
            os_name_ver TEXT,
            latlong TEXT,
            city TEXT,
            state TEXT,
            country TEXT,
            act_name TEXT NOT NULL,
            act_mail TEXT NOT NULL,
            act_mob TEXT NOT NULL,
            Name TEXT NOT NULL,
            Email_ID TEXT NOT NULL,
            resume_score TEXT NOT NULL,
            Timestamp TEXT NOT NULL,
            Page_no TEXT NOT NULL,
            Predicted_Field TEXT NOT NULL,
            User_level TEXT NOT NULL,
            Actual_skills TEXT NOT NULL,
            Recommended_skills TEXT NOT NULL,
            Recommended_courses TEXT NOT NULL,
            pdf_name TEXT NOT NULL
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            feed_name TEXT NOT NULL,
            feed_email TEXT NOT NULL,
            feed_score TEXT NOT NULL,
            comments TEXT,
            Timestamp TEXT NOT NULL
        );
    """)
    connection.commit()

def insert_data(sec_token, ip_add, host_name, dev_user, os_name_ver, latlong,
                city, state, country, act_name, act_mail, act_mob, name, email,
                res_score, timestamp, no_of_pages, reco_field, cand_level,
                skills, recommended_skills, courses, pdf_name):
    if not connection:
        return
    insert_sql = """
        INSERT INTO user_data (
            sec_token, ip_add, host_name, dev_user, os_name_ver, latlong,
            city, state, country, act_name, act_mail, act_mob, Name, Email_ID,
            resume_score, Timestamp, Page_no, Predicted_Field, User_level,
            Actual_skills, Recommended_skills, Recommended_courses, pdf_name
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    rec_values = (sec_token, ip_add, host_name, dev_user, os_name_ver, str(latlong),
                  city, state, country, act_name, act_mail, act_mob, name, email,
                  res_score, timestamp, no_of_pages, reco_field, cand_level,
                  skills, recommended_skills, courses, pdf_name)
    cursor.execute(insert_sql, rec_values)
    connection.commit()

def insertf_data(feed_name, feed_email, feed_score, comments, Timestamp):
    if not connection:
        return
    insertfeed_sql = """
        INSERT INTO user_feedback (feed_name, feed_email, feed_score, comments, Timestamp)
        VALUES (?,?,?,?,?)
    """
    rec_values = (feed_name, feed_email, feed_score, comments, Timestamp)
    cursor.execute(insertfeed_sql, rec_values)
    connection.commit()

# ======================= UI Config & Styles =======================
st.set_page_config(
   page_title="Smart Resume Analyzer – ATS + AI",
   page_icon="📄",
   layout="wide",
)

# Minimal CSS to make cards look nicer
st.markdown(
    """
    <style>
    .card { background: linear-gradient(180deg, #ffffff 0%, #fbfbfd 100%); border-radius:12px; padding:16px; box-shadow: 0 6px 18px rgba(16,24,40,0.06); }
    .muted { color: #6b7280; font-size: 0.95rem; }
    .provider-pill { display:inline-block; padding:6px 10px; border-radius:999px; font-weight:600; }
    .openai { background:linear-gradient(90deg,#f0f9ff,#e6fffa); color:#0f172a; border:1px solid #e6f2ff; }
    .groq { background:linear-gradient(90deg,#fff7ed,#fff1f2); color:#7c2d12; border:1px solid #ffe6d5; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 Smart Resume Analyzer — ATS Screening + AI Coach")
st.caption("Fast, secure resume feedback with semantic job matching, skill-gap analysis and AI coaching (OpenAI → Groq fallback).")

# ======================= Helpers =======================
def get_csv_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def pdf_reader(file_path: str) -> str:
    try:
        return pdf_extract_text(file_path) or ""
    except Exception:
        return ""

def show_pdf(file_path: str):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="650"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception:
        st.info("Preview unavailable; parsing will still run.")

# Optional embeddings loader
@st.cache_resource(show_spinner=False)
def load_embed_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBED_MODEL_NAME)
    except Exception as e:
        st.warning(f"Embedding model not available ({e}). Semantic matching disabled.")
        return None

embed_model = load_embed_model()

def cosine_similarity(a_emb, b_emb):
    try:
        import numpy as np
        a = a_emb / (np.linalg.norm(a_emb) + 1e-9)
        b = b_emb / (np.linalg.norm(b_emb) + 1e-9)
        return float(np.dot(a, b))
    except Exception:
        return 0.0

def semantic_match(resume_text, job_desc):
    if not embed_model:
        return None
    try:
        a = embed_model.encode(resume_text)
        b = embed_model.encode(job_desc)
        sim = cosine_similarity(a, b)
        return round(max(0.0, min(1.0, sim)) * 100.0, 2)
    except Exception:
        return None

# ======================= Skill KB & Gap Analysis =======================
IDEAL_SKILLS = {
    "Data Science": {
        "core": [
            "python","statistics","probability","pandas","numpy","scikit-learn","sql",
            "data visualization","matplotlib","machine learning","deep learning","nlp",
            "tensorflow","pytorch","feature engineering","model evaluation","mlops"
        ],
        "emerging": ["langchain","rag","vector db","llmops","lightgbm","xgboost"]
    },
    "Web Development": {
        "core": [
            "html","css","javascript","react","node.js","express","django","flask",
            "rest api","git","testing","sql","postgresql","mongodb","deployment","docker"
        ],
        "emerging": ["next.js","vite","fastapi","graphql","vercel","cloud run"]
    },
    "Android Development": {
        "core": ["java","kotlin","android","xml","jetpack","sqlite","git","material ui"],
        "emerging": ["flutter","compose","kmp"]
    },
    "IOS Development": {
        "core": ["swift","xcode","uikit","autolayout","cocoapods","sqlite","git"],
        "emerging": ["swiftui","combine"]
    },
    "UI-UX Development": {
        "core": [
            "ux","ui","figma","prototyping","wireframes","user research","information architecture",
            "usability testing","visual design","interaction design"
        ],
        "emerging": ["design systems","motion design","design tokens"]
    }
}

SKILL_KB = sorted(set(
    sum((v["core"] + v["emerging"] for v in IDEAL_SKILLS.values()), [])
    + [
        "typescript","angular","vue","svelte","tailwind","bootstrap","graphql","rest","api",
        "aws","azure","gcp","kubernetes","docker","ci cd","linux","gitlab","tableau","power bi",
        "excel","spark","hadoop","airflow","lightgbm","xgboost","nlp","opencv","computer vision",
        "fastapi","vercel","cloud run","spring","spring boot","wordpress","php","laravel","magento",
        "c#","asp.net","java","flask","django","react js","node","pytorch","keras","tensorflow",
        "streamlit","sqlite","postgresql","mongodb"
    ]
))

def normalize_skill(s: str) -> str:
    return re.sub(r"[^a-z0-9\+\.#]+", " ", (s or "").lower()).strip()

def analyze_gaps(user_skills: List[str], target_field: str):
    target = IDEAL_SKILLS.get(target_field)
    if not target:
        return [], [], []
    u = {normalize_skill(x) for x in user_skills}
    core = {normalize_skill(x) for x in target["core"]}
    emerging = {normalize_skill(x) for x in target["emerging"]}
    missing = sorted(list(core - u))
    covered = sorted(list(core & u))
    emerging_missing = sorted(list(emerging - u))
    return missing, covered, emerging_missing

# ======================= Lightweight Resume Parser =======================

def _pdf_num_pages(pdf_path: str) -> int:
    """
    Robust page-counting:
    1) Try PyPDF2 (preferred)
    2) Fallback to pdfminer.extract_pages
    3) Heuristic: count '/Type /Page' tokens in binary
    4) Final fallback -> return 1 (so single-page resumes aren't misclassified as zero)
    """
    pages = 0
    # Try PyPDF2 / pypdf
    try:
        from PyPDF2 import PdfReader
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            pages = len(reader.pages)
            if pages and pages > 0:
                return pages
    except Exception:
        pass

    # Try pdfminer.extract_pages
    try:
        from pdfminer.high_level import extract_pages
        pages = sum(1 for _ in extract_pages(pdf_path))
        if pages and pages > 0:
            return pages
    except Exception:
        pass

    # Heuristic binary scan for page tokens
    try:
        with open(pdf_path, "rb") as f:
            data = f.read()
            pages = data.count(b"/Type /Page")
            if pages and pages > 0:
                return pages
    except Exception:
        pass

    # Safety fallback to 1 (avoid 0 pages)
    return 1

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?)[-.\s]?)?\d{3,4}[-.\s]?\d{4}")
DEGREE_PATTERNS = [
    r"\b(b\.?tech|b\.?e\.?|bachelor(?:'s)? in [a-z &]+|bsc|bs)\b",
    r"\b(m\.?tech|m\.?e\.?|master(?:'s)? in [a-z &]+|msc|ms|mba)\b",
    r"\b(ph\.?d|doctorate)\b",
]

def extract_email(text: str) -> str:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else ""

def extract_phone(text: str) -> str:
    cands = PHONE_RE.findall(text)
    for c in cands:
        digits = re.sub(r"\D", "", c)
        if 10 <= len(digits) <= 13:
            return c.strip()
    return ""

def extract_degree(text: str) -> str:
    lower = text.lower()
    for pat in DEGREE_PATTERNS:
        m = re.search(pat, lower)
        if m:
            return m.group(0).upper()
    return ""

def extract_name(text: str, email: str, phone: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines[:20]:
        if email and email in l: continue
        if phone and phone in l: continue
        if len(l.split()) <= 5 and re.search(r"[A-Za-z]", l) and len(l) >= 3:
            return " ".join(w.capitalize() for w in l.split())
    return ""

def extract_skills(text: str) -> List[str]:
    text_l = " " + normalize_skill(text) + " "
    found = set()
    for skill in sorted(SKILL_KB, key=lambda s: -len(s)):
        pat = r"\b" + re.escape(normalize_skill(skill)) + r"\b"
        if re.search(pat, text_l):
            found.add(skill)
    dedup = sorted(found, key=lambda s: (len(s), s))
    return dedup

def parse_resume(pdf_path: str) -> Tuple[Dict, str]:
    text = pdf_reader(pdf_path)
    email = extract_email(text)
    phone = extract_phone(text)
    degree = extract_degree(text)
    name = extract_name(text, email, phone)
    pages = _pdf_num_pages(pdf_path)
    skills = extract_skills(text)
    data = {
        "name": name or "-",
        "email": email or "-",
        "mobile_number": phone or "-",
        "degree": degree or "-",
        "no_of_pages": pages or 0,
        "skills": skills or []
    }
    return data, text

# ======================= Course Recommender =======================
def course_recommender(course_list):
    st.subheader("**Courses & Certificates Recommendations 👨‍🎓**")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5, key="num_courses")
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course

# ======================= ATS Red Flags =======================
def detect_red_flags(resume_text):
    flags = []
    if not re.search(r"\b(project|projects)\b", resume_text, re.I):
        flags.append("No **Projects** section found.")
    if not re.search(r"\b(education|qualification|degree)\b", resume_text, re.I):
        flags.append("No **Education** section found.")
    if not re.search(r"\b(20\d{2}|19\d{2})\b", resume_text):
        flags.append("No obvious **years/dates** detected (work/education timelines).")
    long_paras = [p for p in resume_text.split("\n") if len(p) > 600]
    if len(long_paras) >= 1:
        flags.append("Contains **very long paragraphs**; use bullet points for ATS readability.")
    if len(resume_text) < 400:
        flags.append("Resume text seems **very short** — graphics/tables may hurt ATS parsing.")
    return flags

# ======================= AI Helpers (OpenAI -> Groq fallback) =======================
def get_ai_suggestions(prompt_text: str, provider_hint: str = "auto") -> Dict:
    """
    Returns dict: { 'text': str or None, 'provider': 'openai'|'groq'|'local'|'none', 'error': str|None }
    provider_hint: 'openai', 'groq', or 'auto'
    """
    # Local fallback (simple rules) if no provider available
    def local_fallback(text):
        lines = []
        lines.append("Local Suggestions (fallback):")
        if len(text) < 200:
            lines.append("- Resume seems short; add a concise summary and more project bullets with metrics.")
        lines.append("- Use action verbs and quantify impact (e.g. reduced X by 20%).")
        lines.append("- Ensure Skills, Projects, Education sections are present.")
        return "\n".join(lines)

    # Try OpenAI
    if provider_hint in ("openai", "auto") and USE_OPENAI:
        try:
            resp = openai_client.chat.completions.create(
                model=OPENAI_GPT_MODEL,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.2,
                max_tokens=800,
            )
            txt = resp.choices[0].message.content.strip()
            return {"text": txt, "provider": "openai", "error": None}
        except Exception as e:
            # If it's a quota error or other, try groq next
            err_txt = str(e)
            # fall through to try Groq
            provider_err = err_txt

    # Try Groq
    if provider_hint in ("groq", "auto") and USE_GROQ and groq_client:
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.2,
                max_tokens=800,
            )
            # Groq response shape is OpenAI-compatible-ish
            txt = resp.choices[0].message.content.strip()
            return {"text": txt, "provider": "groq", "error": None}
        except Exception as e:
            provider_err = str(e)

    # If both providers missing or failed, return local fallback
    return {"text": local_fallback(prompt_text), "provider": "local", "error": provider_err if 'provider_err' in locals() else None}

# ======================= Sidebar & Nav =======================
st.sidebar.markdown("## Navigation")
activities = ["User", "Feedback", "About", "Admin"]
choice = st.sidebar.selectbox("Choose:", activities)

# ======================= USER FLOW =======================
if choice == "User":
    # Two-column top area for user details + JD
    colA, colB = st.columns([1.2, 1])
    with colA:
        st.subheader("👤 Candidate Info")
        act_name = st.text_input('Name*', value="")
        act_mail = st.text_input('Email*', value="")
        act_mob  = st.text_input('Mobile Number*', value="")
    with colB:
        st.subheader("📋 Optional: Paste Job Description")
        job_desc = st.text_area("Paste full JD here (optional)", height=200)

    # system metadata
    sec_token = secrets.token_urlsafe(16)
    try:
        host_name = socket.gethostname()
        ip_add = socket.gethostbyname(host_name)
        dev_user = os.getlogin()
    except Exception:
        host_name, ip_add, dev_user = "unknown", "0.0.0.0", "unknown"

    os_name_ver = f"{platform.system()} {platform.release()}"

    # Non-blocking geo (fast-fail)
    latlong, city, state, country = None, "", "", ""
    if GEO_OK:
        try:
            g = geocoder.ip('me', timeout=2.0)
            latlong = g.latlng if g else None
            if latlong:
                geolocator = Nominatim(user_agent="resume_ai_app", timeout=2)
                location = geolocator.reverse(latlong, language='en')
                address = location.raw.get('address', {}) if location else {}
                city = address.get('city', '') or address.get('town','') or address.get('village','')
                state = address.get('state','')
                country = address.get('country','')
        except Exception:
            pass

    st.markdown("### Upload Your Resume (PDF)")
    pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])

    if pdf_file is not None:
        with st.spinner('Analyzing…'):
            time.sleep(0.6)

        save_dir = './Uploaded_Resumes'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, pdf_file.name)
        pdf_name = pdf_file.name
        with open(save_path, "wb") as f:
            f.write(pdf_file.getbuffer())

        show_pdf(save_path)

        # Parse resume
        resume_data, resume_text = parse_resume(save_path)

        if resume_data:
            # Header summary
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            hcol1, hcol2, hcol3 = st.columns([2,1,1])
            with hcol1:
                st.header(f"{resume_data.get('name','-')}")
                st.markdown(f"<div class='muted'>{resume_data.get('degree','-')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='muted'>{resume_data.get('email','-')} · {resume_data.get('mobile_number','-')}</div>", unsafe_allow_html=True)
            with hcol2:
                st.metric("Pages", resume_data.get('no_of_pages',0))
                st.metric("Detected Skills", len(resume_data.get('skills',[])))
            with hcol3:
                provider_html = ""
                if USE_OPENAI:
                    provider_html += "<span class='provider-pill openai'>OpenAI ready</span><br>"
                if USE_GROQ:
                    provider_html += "<span class='provider-pill groq'>Groq ready</span>"
                if not (USE_OPENAI or USE_GROQ):
                    provider_html = "<span class='muted'>AI providers not configured</span>"
                st.markdown(provider_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Experience Level
            pages = resume_data.get('no_of_pages', 0) or 0
            cand_level = "Fresher"
            if pages < 1:
                cand_level = "NA"
                st.info("Resume has no pages detected — check PDF.")
            elif any(k in resume_text.upper() for k in ['INTERNSHIP', 'INTERNSHIPS']):
                cand_level = "Intermediate"
            elif any(k in resume_text.upper() for k in ['EXPERIENCE', 'WORK EXPERIENCE']):
                cand_level = "Experienced"

            # Skills display
            st.subheader("🔧 Skills detected")
            extracted_skills = resume_data.get('skills') or []
            if HAS_ST_TAGS:
                st_tags(label='Your Current Skills', text='Detected from resume', value=extracted_skills, key='skills_current')
            else:
                st.write(", ".join(extracted_skills) if extracted_skills else "No skills detected.")

            # Field detection (heuristic)
            st.subheader("🏷️ Field prediction & recommended skills")
            ds_keyword = ['tensorflow','keras','pytorch','machine learning','deep learning','flask','streamlit','scikit-learn','nlp']
            web_keyword = ['react', 'django', 'node js','node','php','laravel','magento','wordpress','javascript', 'angular', 'c#', 'asp.net', 'flask','html','css']
            android_keyword = ['android','android development','flutter','kotlin','xml','kivy','java']
            ios_keyword = ['ios','ios development','swift','cocoa','cocoa touch','xcode','objective-c']
            uiux_keyword = ['ux','adobe xd','figma','zeplin','balsamiq','ui','prototyping','wireframes','storyframes','adobe photoshop','photoshop','illustrator','after effects','premier pro','indesign','user research']
            n_any = ['english','communication','writing','microsoft office','leadership','customer management','social media']

            reco_field = ''
            rec_course = []
            recommended_skills = []
            field_detected = False

            for s in extracted_skills:
                l = (s or "").lower().strip()
                if l in ds_keyword:
                    reco_field = 'Data Science'
                    recommended_skills = ['Data Visualization','Predictive Analysis','Statistical Modeling','Data Mining','Clustering & Classification','Data Analytics','ML Algorithms','Scikit-learn','Tensorflow','Pytorch']
                    rec_course = course_recommender(ds_course)
                    field_detected = True
                    break
                elif l in web_keyword:
                    reco_field = 'Web Development'
                    recommended_skills = ['React','Django','Node JS','REST APIs','Docker','Testing']
                    rec_course = course_recommender(web_course)
                    field_detected = True
                    break
                elif l in android_keyword:
                    reco_field = 'Android Development'
                    recommended_skills = ['Kotlin','Jetpack','Android SDK','SQLite','GIT']
                    rec_course = course_recommender(android_course)
                    field_detected = True
                    break
                elif l in ios_keyword:
                    reco_field = 'IOS Development'
                    recommended_skills = ['Swift','Xcode','UIKit','Auto-Layout','SwiftUI']
                    rec_course = course_recommender(ios_course)
                    field_detected = True
                    break
                elif l in uiux_keyword:
                    reco_field = 'UI-UX Development'
                    recommended_skills = ['Figma','Prototyping','Wireframes','User Research']
                    rec_course = course_recommender(uiux_course)
                    field_detected = True
                    break
                elif l in n_any:
                    reco_field = 'NA'
                    recommended_skills = ['No Recommendations']
                    rec_course = []
                    field_detected = True
                    break

            if not field_detected:
                st.info("Couldn’t confidently detect a field from skills — use Semantic JD Match or AI below.")

            if reco_field:
                c1, c2 = st.columns([2,1])
                with c1:
                    st.markdown(f"**Predicted field:** {reco_field}")
                    st.markdown("**Recommended skills:**")
                    if HAS_ST_TAGS:
                        st_tags(label='', text='', value=recommended_skills, key='reco_skills')
                    else:
                        st.write(", ".join(recommended_skills) if recommended_skills else "—")
                with c2:
                    st.markdown("**Courses**")
                    if isinstance(rec_course, list) and rec_course:
                        for i, c in enumerate(rec_course, start=1):
                            st.markdown(f"{i}. {c}")
                    elif isinstance(rec_course, str):
                        st.markdown(rec_course)
                    else:
                        st.markdown("No course recommendations available.")

            # Gap analysis
            if reco_field and reco_field != 'NA':
                st.subheader("🧭 Gap Analysis vs Ideal Skill Profile")
                miss, covered, emerg = analyze_gaps(extracted_skills, reco_field)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Missing Core Skills:**")
                    st.write(", ".join(miss) if miss else "Looks great. No critical gaps.")
                with c2:
                    st.markdown("**Covered Core Skills:**")
                    st.write(", ".join(covered) if covered else "Consider covering a few core skills from the list.")
                with c3:
                    st.markdown("**Emerging (Future-proof):**")
                    st.write(", ".join(emerg) if emerg else "You're up-to-date!")

            # Semantic JD matching
            st.subheader("🔎 Semantic Match to Job Description (optional)")
            if job_desc and len(job_desc.strip()) > 20:
                score = semantic_match(resume_text, job_desc)
                if score is not None:
                    st.success(f"**JD Match Score:** {score}%")
                    st.progress(min(100, int(score)))
                else:
                    st.warning("Semantic model unavailable — install `sentence-transformers` to enable embeddings.")
            else:
                st.caption("Paste a JD above to compute semantic match.")

            # ATS Red flags
            st.subheader("🚩 ATS Red-Flag Checks")
            flags = detect_red_flags(resume_text)
            if flags:
                for f in flags:
                    st.warning(f"- {f}")
            else:
                st.success("No major ATS red flags detected.")

            # ----- GPT / Groq Suggestions (visual upgrade) -----
            st.subheader("🧠 AI Suggestions (OpenAI → Groq fallback)")
            prompt = f"""
You are an expert ATS coach. Given this resume text, suggest the top 6 improvements.
Focus on:
- impact bullets (metrics, action verbs),
- missing critical sections,
- tailoring tips for the candidate's target field: {reco_field or 'Unknown'},
- phrasing that is ATS-friendly and concise.

Resume:
\"\"\"{resume_text[:12000]}\"\"\"
Provide suggestions as numbered bullets and short example rewrites where useful.
"""
            with st.spinner("Generating AI suggestions (tries OpenAI, falls back to Groq, then local heuristics)…"):
                ai_resp = get_ai_suggestions(prompt, provider_hint="auto")

            provider = ai_resp.get("provider", "none")
            ai_text = ai_resp.get("text", "")
            err = ai_resp.get("error", None)

            # Display provider badge
            provider_badge = ""
            if provider == "openai":
                provider_badge = "<span class='provider-pill openai'>Answer by OpenAI</span>"
            elif provider == "groq":
                provider_badge = "<span class='provider-pill groq'>Answer by Groq</span>"
            elif provider == "local":
                provider_badge = "<span class='provider-pill' style='background:#f3f4f6;color:#374151'>Local fallback</span>"
            st.markdown(provider_badge, unsafe_allow_html=True)

            # AI output card
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if ai_text:
                st.markdown(ai_text)
                # copy/download actions
                st.download_button("⬇️ Download suggestions (.txt)", data=ai_text, file_name=f"{pdf_name}_suggestions.txt")
                if st.button("Copy suggestions to clipboard (copy visible text)"):
                    st.write("Select the suggestions and copy manually; Streamlit can't access clipboard programmatically in all browsers.")
            else:
                st.info("AI couldn't generate suggestions. See fallback suggestions below.")
                if err:
                    st.warning(f"Provider error: {err}")
            st.markdown("</div>", unsafe_allow_html=True)

            # ----- Resume Score (heuristic) -----
            st.subheader("✍️ Resume Tips & Heuristic Score")
            resume_score = 0
            if re.search(r"\b(objective|summary)\b", resume_text, re.I):
                resume_score += 6
                st.markdown("<h5 style='color:#10b981;'>[+] Objective/Summary present</h5>", unsafe_allow_html=True)
            else:
                st.markdown("<h5>[-] Add a concise objective/summary</h5>", unsafe_allow_html=True)

            if re.search(r"\b(education|school|college|university)\b", resume_text, re.I):
                resume_score += 12
                st.markdown("<h5 style='color:#10b981;'>[+] Education section present</h5>", unsafe_allow_html=True)
            else:
                st.markdown("<h5>[-] Add Education details</h5>", unsafe_allow_html=True)

            if re.search(r"\b(experience|work experience|internship|internships)\b", resume_text, re.I):
                resume_score += 16
                st.markdown("<h5 style='color:#10b981;'>[+] Experience/Internships present</h5>", unsafe_allow_html=True)
            else:
                st.markdown("<h5>[-] Add Experience/Internships</h5>", unsafe_allow_html=True)

            if re.search(r"\b(skill|skills)\b", resume_text, re.I):
                resume_score += 7
                st.markdown("<h5 style='color:#10b981;'>[+] Skills section present</h5>", unsafe_allow_html=True)
            else:
                st.markdown("<h5>[-] Add a dedicated Skills section</h5>", unsafe_allow_html=True)

            if re.search(r"\b(hobby|hobbies)\b", resume_text, re.I):
                resume_score += 4
                st.markdown("<h5 style='color:#10b981;'>[+] Hobbies present</h5>", unsafe_allow_html=True)

            if re.search(r"\b(interest|interests)\b", resume_text, re.I):
                resume_score += 5
                st.markdown("<h5 style='color:#10b981;'>[+] Interests present</h5>", unsafe_allow_html=True)

            if re.search(r"\b(achievement|achievements|awards)\b", resume_text, re.I):
                resume_score += 13
                st.markdown("<h5 style='color:#10b981;'>[+] Achievements present</h5>", unsafe_allow_html=True)

            if re.search(r"\b(project|projects)\b", resume_text, re.I):
                resume_score += 19
                st.markdown("<h5 style='color:#10b981;'>[+] Projects present</h5>", unsafe_allow_html=True)
            else:
                st.markdown("<h5>[-] Add Projects with metrics</h5>", unsafe_allow_html=True)

            st.subheader("Resume Score")
            my_bar = st.progress(0)
            for pct in range(min(resume_score, 100)):
                time.sleep(0.004)
                my_bar.progress(pct + 1)
            st.success(f"** Your Resume Writing Score: {min(resume_score,100)} **")
            st.warning("Note: Score is heuristic & surface-level (sections & density).")

            # Timestamp and persist
            ts = time.time()
            cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            timestamp = f"{cur_date}_{cur_time}"

            try:
                insert_data(
                    str(sec_token), str(ip_add), host_name, dev_user, os_name_ver, str(latlong),
                    city, state, country, act_name, act_mail, act_mob,
                    resume_data.get('name','-'), resume_data.get('email','-'),
                    str(resume_score), timestamp, str(resume_data.get('no_of_pages','-')),
                    reco_field or 'NA', cand_level, json.dumps(extracted_skills),
                    json.dumps(recommended_skills), json.dumps(rec_course), pdf_name
                )
            except Exception as e:
                st.info(f"Skipped DB save: {e}")

            # Videos + celebration
            st.markdown("---")
            st.header("Bonus: Quick tips videos")
            vcol1, vcol2 = st.columns(2)
            with vcol1:
                if resume_videos:
                    st.video(random.choice(resume_videos))
            with vcol2:
                if interview_videos:
                    st.video(random.choice(interview_videos))
            st.balloons()

        else:
            st.error('Something went wrong while parsing your resume.')

# ======================= FEEDBACK FLOW =======================
elif choice == 'Feedback':
    ts = time.time()
    cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    timestamp = f"{cur_date}_{cur_time}"

    with st.form("feedback_form"):
        st.write("Feedback form")
        feed_name = st.text_input('Name')
        feed_email = st.text_input('Email')
        feed_score = st.slider('Rate Us From 1 - 5', 1, 5)
        comments = st.text_input('Comments')
        submitted = st.form_submit_button("Submit")
        if submitted:
            try:
                insertf_data(feed_name, feed_email, feed_score, comments, timestamp)
                st.success("Thanks! Your Feedback was recorded.")
                st.balloons()
            except Exception as e:
                st.error(f"Could not save feedback: {e}")

    if connection:
        try:
            plotfeed_data = pd.read_sql('select * from user_feedback', connection)
            if not plotfeed_data.empty:
                labels = plotfeed_data.feed_score.unique()
                values = plotfeed_data.feed_score.value_counts()
                st.subheader("Past User Ratings")
                fig = px.pie(values=values, names=labels, title="User Rating Score (1 - 5)")
                st.plotly_chart(fig, use_container_width=True)

                cursor.execute('select feed_name, comments from user_feedback')
                plfeed_cmt_data = cursor.fetchall()
                st.subheader("User Comments")
                dff = pd.DataFrame(plfeed_cmt_data, columns=['User', 'Comment'])
                st.dataframe(dff, use_container_width=True)
        except Exception as e:
            st.info(f"No feedback data yet ({e}).")

# ======================= ABOUT FLOW =======================
elif choice == 'About':
    st.subheader("About — Smart Resume Analyzer")
    st.markdown("""
Smart Resume Analyzer is an AI Career Assistant.

**How it works**
- Parsing: PDF → structured fields, skills, and text (PDFMiner + regex).
- Scoring: Heuristics for ATS sections & content density.
- Gap Analysis: Compare your skills to ideal profiles across fields.
- Semantic Match: Sentence-BERT embeddings for JD ↔ Resume matching (optional).
- AI Suggestions: OpenAI (preferred) → Groq (fallback) → Local heuristics.
- Analytics: Admin dashboard for trends & insights.

**Tech stack**
- Frontend: Streamlit
- DB: SQLite
- Optional: OpenAI / Groq / Sentence-BERT
""")

# ======================= ADMIN FLOW (no auth) =======================
else:
    st.success('Admin Dashboard')
    st.caption("Recruiter Intelligence Dashboard — track candidate skills, gaps, trends.")

    if not connection:
        st.error("Database not connected.")
    else:
        if st.button('Open Admin Dashboard'):
            try:
                plot_df = pd.read_sql("""
                    SELECT ID, ip_add, resume_score, Predicted_Field, User_level, city, state, country
                    FROM user_data
                """, connection)
                st.success(f"Welcome Admin! Total {len(plot_df)} users have used the tool.")

                data = pd.read_sql("""
                    SELECT ID, sec_token, ip_add, act_name, act_mail, act_mob, Predicted_Field, Timestamp,
                           Name, Email_ID, resume_score, Page_no, pdf_name, User_level,
                           Actual_skills, Recommended_skills, Recommended_courses, city, state, country,
                           latlong, os_name_ver, host_name, dev_user
                    FROM user_data
                """, connection)

                st.header("Users Data")
                st.dataframe(data, use_container_width=True)
                st.markdown(get_csv_download_link(data,'User_Data.csv','Download Report'), unsafe_allow_html=True)

                feed_df = pd.read_sql('select * from user_feedback', connection)
                if not feed_df.empty:
                    st.subheader("User Ratings")
                    labels = feed_df.feed_score.unique()
                    values = feed_df.feed_score.value_counts()
                    fig = px.pie(values=values, names=labels, title="User Rating Score (1-5)")
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Predicted Field Distribution")
                if not plot_df.empty:
                    values = plot_df.Predicted_Field.value_counts()
                    fig = px.pie(values=values, names=values.index, title='Predicted Field according to Skills')
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("User Experience Level")
                if not plot_df.empty:
                    values = plot_df.User_level.value_counts()
                    fig = px.pie(values=values, names=values.index, title="User Experience Level")
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Resume Score Distribution")
                if not plot_df.empty:
                    values = plot_df.resume_score.value_counts()
                    fig = px.bar(x=values.index, y=values.values, labels={'x':'Score','y':'Count'}, title='Resume Scores')
                    st.plotly_chart(fig, use_container_width=True)

                geo_cols = st.columns(3)
                with geo_cols[0]:
                    st.subheader("By City")
                    if not plot_df.empty:
                        vals = plot_df.city.value_counts().head(15)
                        st.bar_chart(vals)
                with geo_cols[1]:
                    st.subheader("By State")
                    if not plot_df.empty:
                        vals = plot_df.state.value_counts().head(15)
                        st.bar_chart(vals)
                with geo_cols[2]:
                    st.subheader("By Country")
                    if not plot_df.empty:
                        vals = plot_df.country.value_counts().head(15)
                        st.bar_chart(vals)

                st.header("🔥 Trend Insights — Top Missing Core Skills")
                rows = cursor.execute('SELECT Actual_skills, Predicted_Field FROM user_data').fetchall()
                missing_counter = {}
                for skills_json, field in rows:
                    try:
                        skills_list = json.loads(skills_json)
                    except Exception:
                        try:
                            skills_list = ast.literal_eval(skills_json)
                        except Exception:
                            skills_list = []
                    field = field or "NA"
                    if field in IDEAL_SKILLS:
                        miss, _, _ = analyze_gaps(skills_list, field)
                        for m in miss:
                            missing_counter[m] = missing_counter.get(m, 0) + 1

                if missing_counter:
                    miss_df = pd.DataFrame(
                        sorted(missing_counter.items(), key=lambda x: x[1], reverse=True)[:20],
                        columns=["Skill","Missing Count"]
                    )
                    st.dataframe(miss_df, use_container_width=True)
                    fig = px.bar(miss_df, x="Skill", y="Missing Count", title="Top Missing Skills", height=450)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data to compute missing-skill trends yet.")

            except Exception as e:
                st.error(f"Admin dashboard error: {e}")
