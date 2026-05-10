import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ---------- Function to Extract Text from PDF ----------
def extract_text(pdf_file):
    text = ""

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

    return text


# ---------- Function to Clean Text ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text


# ---------- Streamlit UI ----------
st.title("🤖 AI Resume Screener")

st.write("Upload Resume and Job Description")

resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

job_description = st.text_area("Paste Job Description")


if resume_file and job_description:

    # Extract Resume Text
    resume_text = extract_text(resume_file)

    # Clean Text
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(job_description)

    # TF-IDF Vectorization
    documents = [resume_clean, jd_clean]

    tfidf = TfidfVectorizer()

    tfidf_matrix = tfidf.fit_transform(documents)

    # Cosine Similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    match_score = round(similarity[0][0] * 100, 2)

    # Display Result
    st.subheader("📊 Match Score")
    st.success(f"{match_score}% Match Found")

    # Skill Matching
    skills = [
        "python", "sql", "machine learning",
        "deep learning", "excel", "power bi",
        "tableau", "nlp", "data analysis"
    ]

    matched_skills = []

    for skill in skills:
        if skill in resume_clean:
            matched_skills.append(skill)

    st.subheader("✅ Skills Found")

    if matched_skills:
        st.write(matched_skills)
    else:
        st.write("No matching skills found.")