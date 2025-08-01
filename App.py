import streamlit as st
import nltk
import spacy
import re
import difflib
import io
nltk.download('stopwords', quiet=True)
# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    raise

import pandas as pd
import base64, random, time, datetime
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter
from streamlit_tags import st_tags
import sqlite3
import plotly.express as px

from Courses import ds_course, web_course, android_course, ios_course, uiux_course

DB_PATH = "sra.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_data (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT NOT NULL,
            Email_ID TEXT NOT NULL,
            resume_score TEXT NOT NULL,
            Timestamp TEXT NOT NULL,
            Page_no TEXT NOT NULL,
            Predicted_Field TEXT NOT NULL,
            User_level TEXT NOT NULL,
            Actual_skills TEXT NOT NULL,
            Recommended_skills TEXT NOT NULL,
            Recommended_courses TEXT NOT NULL
        );
    """)
    conn.commit()
    return conn, cursor


def insert_data(cursor, conn, name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, skills,
                recommended_skills, courses):
    insert_sql = """
        INSERT INTO user_data
        (Name, Email_ID, resume_score, Timestamp, Page_no, Predicted_Field, User_level, Actual_skills, Recommended_skills, Recommended_courses)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    rec_values = (
        name, email, str(res_score), timestamp, str(no_of_pages), reco_field, cand_level, skills,
        recommended_skills, courses
    )
    cursor.execute(insert_sql, rec_values)
    conn.commit()


def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


def pdf_reader(file_obj_or_path):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    if isinstance(file_obj_or_path, (io.BytesIO, io.BufferedReader)):
        fh = file_obj_or_path
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
    else:
        with open(file_obj_or_path, 'rb') as fh:
            for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                page_interpreter.process_page(page)
    text = fake_file_handle.getvalue()
    converter.close()
    fake_file_handle.close()
    return text


def simple_resume_parser(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    name = ""
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    email = email_match.group(0) if email_match else ""
    phone_match = re.search(r'(\+?\d[\d\-\s]{6,}\d)', text)
    phone = phone_match.group(0) if phone_match else ""
    if lines:
        first_line = lines[0]
        if len(first_line.split()) >= 2 and any(w[0].isupper() for w in first_line.split() if w):
            name = first_line
    if not name and email:
        prefix = email.split("@")[0]
        parts = re.split(r'[._\-]', prefix)
        name = " ".join(p.capitalize() for p in parts if p)
    if not name and lines:
        name = lines[0]
    all_keywords = [
        'tensorflow', 'keras', 'pytorch', 'machine learning', 'deep learning', 'flask', 'streamlit',
        'react', 'django', 'node js', 'react js', 'php', 'laravel', 'magento', 'wordpress',
        'javascript', 'angular js', 'c#', 'android', 'flutter', 'kotlin', 'ios', 'swift', 'figma',
        'adobe xd', 'user experience', 'ux', 'ui'
    ]
    found = set()
    lowered = text.lower()
    for kw in all_keywords:
        if kw in lowered:
            found.add(kw)
    skills = list(found)
    return {
        'name': name,
        'email': email,
        'mobile_number': phone,
        'skills': skills,
        'no_of_pages': 1
    }


def classify_skills(skills_list):
    category_keywords = {
        "Data Science": ['tensorflow', 'keras', 'pytorch', 'machine learning', 'deep learning', 'flask', 'streamlit', 'data science'],
        "Web Development": ['react', 'django', 'node js', 'react js', 'php', 'laravel', 'magento', 'wordpress', 'javascript', 'angular js', 'c#', 'flask'],
        "Android Development": ['android', 'android development', 'flutter', 'kotlin', 'xml', 'kivy'],
        "IOS Development": ['ios', 'ios development', 'swift', 'cocoa', 'xcode'],
        "UI-UX Development": ['ux', 'user experience', 'adobe xd', 'figma', 'zeplin', 'wireframes', 'prototyping', 'ui']
    }
    scores = {cat: 0 for cat in category_keywords}
    for skill in skills_list:
        low = skill.lower()
        for cat, keywords in category_keywords.items():
            best_ratio = 0
            for kw in keywords:
                ratio = difflib.SequenceMatcher(None, low, kw).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
            if best_ratio >= 0.75:
                scores[cat] += 1
    best_cat = max(scores, key=lambda k: scores[k])
    total = sum(scores.values()) or 1
    confidence = round(100 * scores[best_cat] / total)
    return best_cat, scores, confidence


def detect_intent_from_text(text):
    target = "data science"
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    bigrams = [' '.join([words[i], words[i+1]]) for i in range(len(words)-1)]
    for bg in bigrams:
        if difflib.SequenceMatcher(None, bg, target).ratio() >= 0.8:
            return "Data Science"
    return None


def extract_jd_keywords(text, top_n=40):
    doc = nlp(text.lower())
    candidates = set()
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        if 1 <= len(chunk_text.split()) <= 3:
            candidates.add(chunk_text)
    for ent in doc.ents:
        ent_text = ent.text.strip()
        if 1 <= len(ent_text.split()) <= 3:
            candidates.add(ent_text)
    for token in doc:
        if not token.is_stop and token.is_alpha and len(token.text) > 2:
            candidates.add(token.lemma_)
    return list(candidates)[:top_n]


def compute_ats_match(jd_keywords, resume_skills):
    matched = set()
    resume_lower = [s.lower() for s in resume_skills]
    for kw in jd_keywords:
        low = kw.lower()
        for skill in resume_lower:
            ratio = difflib.SequenceMatcher(None, low, skill).ratio()
            if ratio >= 0.75 or low in skill or skill in low:
                matched.add(kw)
                break
    missing = set(jd_keywords) - matched
    coverage = int(100 * len(matched) / len(jd_keywords)) if jd_keywords else 0
    return sorted(matched), sorted(missing), coverage


def course_recommender(course_list):
    st.subheader("**Courses & Certificates 🎓 Recommendations**")
    rec_course = []
    no_of_reco = st.slider('How many courses would you like?', 1, 10, 3)
    random.shuffle(course_list)
    for idx, (c_name, c_link) in enumerate(course_list, start=1):
        st.markdown(f"**{idx}.** [{c_name}]({c_link})")
        rec_course.append(c_name)
        if idx == no_of_reco:
            break
    return rec_course


# ✅ ONLY CHANGE: Updated color for Predicted Field box
st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")
st.markdown("""
    <style>
    .card { background: #ffffff; padding: 1rem 1.25rem; border-radius: 16px; box-shadow: 0 20px 40px rgba(0,0,0,0.08); margin-bottom:1rem; }
    .section-title { font-size:1.7rem; font-weight:700; margin-bottom:0.3rem; }
    .badge { background: linear-gradient(135deg,#6e8efb,#a777e3); color:white; padding:5px 12px; border-radius:999px; display:inline-block; font-size:0.75rem; }
    .field-box { background:#4b2be3; color:white; padding:8px 14px; border-radius:12px; display:inline-block; font-weight:600; }
    </style>
""", unsafe_allow_html=True)


def run():
    st.title("🔍 Smart Resume Analyzer")
    st.markdown("## Tailored résumé insights — skills, field inference, job-fit, and writing feedback")
    st.sidebar.markdown("## Role")
    choice = st.sidebar.selectbox("Select Mode:", ["Normal User", "Admin"])

    conn, cursor = init_db()

    if choice == "Normal User":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Upload & Analyze Resume</div>", unsafe_allow_html=True)
        pdf_file = st.file_uploader("📄 Upload your resume (PDF)", type=["pdf"])
        if pdf_file:
            resume_bytes = io.BytesIO(pdf_file.getbuffer())
            resume_text = pdf_reader(resume_bytes)

            try:
                from pyresparser import ResumeParser
                temp_path = f"temp_{int(time.time())}.pdf"
                with open(temp_path, "wb") as f:
                    f.write(resume_bytes.getbuffer())
                resume_data = ResumeParser(temp_path).get_extracted_data()
                if not resume_data:
                    raise ValueError("Empty parse")
                if not resume_data.get('name') and resume_data.get('email'):
                    prefix = resume_data['email'].split('@')[0]
                    parts = re.split(r'[._\-]', prefix)
                    resume_data['name'] = " ".join(p.capitalize() for p in parts if p)
            except Exception:
                resume_data = simple_resume_parser(resume_text)

            # Basic Info
            st.markdown("<div class='section-title'>Resume Analysis</div>", unsafe_allow_html=True)
            name = resume_data.get('name', 'Candidate')
            st.success(f"Hello, {name}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Name**")
                st.markdown(f"<div class='badge'>{name}</div>", unsafe_allow_html=True)
                st.markdown("**Email**")
                st.markdown(f"<div class='badge'>{resume_data.get('email','-')}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("**Contact**")
                st.markdown(f"<div class='badge'>{resume_data.get('mobile_number','-')}</div>", unsafe_allow_html=True)
                st.markdown("**Pages**")
                st.markdown(f"<div class='badge'>{resume_data.get('no_of_pages','-')}</div>", unsafe_allow_html=True)
            with col3:
                pages = resume_data.get('no_of_pages', 0)
                cand_level = "Unknown"
                level_color = "#888"
                if pages == 1:
                    cand_level = "Fresher"; level_color = "#d73b5c"
                elif pages == 2:
                    cand_level = "Intermediate"; level_color = "#1ed760"
                elif pages >= 3:
                    cand_level = "Experienced"; level_color = "#fba171"
                st.markdown("**Candidate Level**")
                st.markdown(f"<div style='background:{level_color}; padding:6px 14px; border-radius:12px; color:white; display:inline-block;'>{cand_level}</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("🧠 Field & Skills Inference")
            detected_skills = resume_data.get('skills', [])
            st_tags(label='Skills detected', text='Extracted from resume', value=detected_skills, key='skills_input')

            reco_field, raw_scores, confidence = classify_skills(detected_skills)
            intent_override = detect_intent_from_text(resume_text)
            if intent_override:
                reco_field = intent_override
                confidence = max(confidence, 85)

            st.markdown(f"**Predicted Field:** <span class='field-box'>{reco_field}</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence (approx):** {confidence}%")

            breakdown_cols = st.columns(len(raw_scores))
            for idx, (cat, score) in enumerate(raw_scores.items()):
                pct = int(100 * score / (sum(raw_scores.values()) or 1))
                with breakdown_cols[idx]:
                    st.markdown(f"**{cat}**")
                    st.progress(pct)
                    st.caption(f"{score} match{'es' if score != 1 else ''}")

            # ==============================
            # ✅ JOB DESCRIPTION / ATS CHECK
            # ==============================
            st.markdown("---")
            st.subheader("📝 Job Description / Role Fit (ATS-style check)")
            jd_choice = st.radio("Provide the job description by:", ["Paste text", "Upload file"], horizontal=True)
            jd_text = ""
            if jd_choice == "Paste text":
                jd_text = st.text_area("Paste the job description here", height=180)
            else:
                jd_file = st.file_uploader("Upload JD (txt or pdf)", type=["txt", "pdf"], key="jd_upload")
                if jd_file:
                    if jd_file.type == "text/plain":
                        jd_text = jd_file.read().decode('utf-8', errors='ignore')
                    elif jd_file.type == "application/pdf":
                        temp_io = io.BytesIO(jd_file.getbuffer())
                        jd_text = pdf_reader(temp_io)

            if jd_text:
                jd_keywords = extract_jd_keywords(jd_text, top_n=40)
                matched, missing, coverage = compute_ats_match(jd_keywords, detected_skills)

                st.markdown(f"**ATS Match Score:** <span style='color:#4b2be3; font-weight:700;'>{coverage}%</span>", unsafe_allow_html=True)
                two = st.columns(2)
                with two[0]:
                    st.markdown("**✅ Matched Keywords**")
                    if matched:
                        for k in matched:
                            st.markdown(f"- {k}")
                    else:
                        st.markdown("_None detected_")
                with two[1]:
                    st.markdown("**⚠️ Missing / Suggested Keywords**")
                    if missing:
                        for k in missing:
                            st.markdown(f"- {k}")
                    else:
                        st.markdown("_Great coverage!_")

                if coverage >= 75:
                    st.success("Your resume aligns well with this job description.")
                elif coverage >= 40:
                    st.warning("Partial overlap. Consider adding some of the missing keywords/skills.")
                else:
                    st.error("Low overlap. Rework your resume to better match the job description.")
            else:
                st.info("Provide a job description to get an ATS-style fit analysis.")

            # ==============================
            # ✅ RECOMMENDATIONS SECTION
            # ==============================
            st.markdown("---")
            st.subheader("🚀 Recommendations")
            recommended_skills = []
            rec_course = []
            if reco_field == "Data Science":
                recommended_skills = ['Data Visualization', 'Predictive Analysis', 'Statistical Modeling',
                                      'Data Mining', 'Clustering & Classification', 'Data Analytics',
                                      'Quantitative Analysis', 'Web Scraping', 'ML Algorithms', 'Keras',
                                      'Pytorch', 'Probability', 'Scikit-learn', 'Tensorflow', "Flask",
                                      'Streamlit']
                st_tags(label='Recommended skills', text='Improve your profile', value=recommended_skills, key='rec_ds')
                rec_course = course_recommender(ds_course)
            elif reco_field == "Web Development":
                recommended_skills = ['React', 'Django', 'Node JS', 'React JS', 'php', 'laravel', 'Magento',
                                      'wordpress', 'Javascript', 'Angular JS', 'c#', 'Flask', 'SDK']
                st_tags(label='Recommended skills', text='Improve your profile', value=recommended_skills, key='rec_web')
                rec_course = course_recommender(web_course)
            elif reco_field == "Android Development":
                recommended_skills = ['Android', 'Android development', 'Flutter', 'Kotlin', 'XML', 'Java',
                                      'Kivy', 'GIT', 'SDK', 'SQLite']
                st_tags(label='Recommended skills', text='Improve your profile', value=recommended_skills, key='rec_android')
                rec_course = course_recommender(android_course)
            elif reco_field == "IOS Development":
                recommended_skills = ['IOS', 'IOS Development', 'Swift', 'Cocoa', 'Cocoa Touch', 'Xcode',
                                      'Objective-C', 'SQLite', 'Plist', 'StoreKit', "UI-Kit", 'AV Foundation',
                                      'Auto-Layout']
                st_tags(label='Recommended skills', text='Improve your profile', value=recommended_skills, key='rec_ios')
                rec_course = course_recommender(ios_course)
            elif reco_field == "UI-UX Development":
                recommended_skills = ['UI-UX', 'Adobe XD', 'Figma', 'Prototyping', 'Wireframes', 'User Experience',
                                      'Illustrator', 'InDesign', 'Photoshop', 'After Effects', 'Premier Pro', 'Lightroom']
                st_tags(label='Recommended skills', text='Improve your profile', value=recommended_skills, key='rec_uiux')
                rec_course = course_recommender(uiux_course)

            # ==============================
            # ✅ RESUME WRITING FEEDBACK
            # ==============================
            st.markdown("---")
            st.subheader("✍️ Resume Writing Feedback")
            feedback_points = []
            if not re.search(r"(objective|summary)", resume_text, re.IGNORECASE):
                feedback_points.append("Consider adding a **Career Objective** or Summary section at the top.")
            if not re.search(r"(project|experience)", resume_text, re.IGNORECASE):
                feedback_points.append("Include **Projects** or Work Experience to strengthen your profile.")
            if len(resume_text.split()) < 150:
                feedback_points.append("Your resume appears quite short — consider adding more details.")
            if feedback_points:
                for fb in feedback_points:
                    st.markdown(f"- {fb}")
            else:
                st.success("Your resume looks comprehensive — great work!")

            # ==============================
            # ✅ STORE DATA IN DATABASE
            # ==============================
            insert_data(cursor, conn,
                        name,
                        resume_data.get('email', '-'),
                        random.randint(60, 99),
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        resume_data.get('no_of_pages', 1),
                        reco_field,
                        cand_level,
                        ", ".join(detected_skills),
                        ", ".join(recommended_skills),
                        ", ".join(rec_course)
                        )

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.title("📊 Admin Dashboard")
        df = pd.read_sql_query("SELECT * FROM user_data", conn)
        st.dataframe(df)
        st.markdown(get_table_download_link(df, 'user_data.csv', 'Download CSV'), unsafe_allow_html=True)


if __name__ == "__main__":
    run()
