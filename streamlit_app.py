from dotenv import load_dotenv
load_dotenv()

import base64
import streamlit as st
import os
import io
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input_prompt, pdf_content, user_input=None, question=None):
    model = genai.GenerativeModel('gemini-1.5-flash')
    if question:
        response = model.generate_content([input_prompt, pdf_content[0], question])
    else:
        response = model.generate_content([input_prompt, pdf_content[0], user_input])
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()
        return [{
            "mime_type": "image/jpeg",
            "data": base64.b64encode(img_byte_arr).decode()
        }]
    raise FileNotFoundError("No file uploaded")

# Streamlit Interface
st.set_page_config(page_title="ATS Resume Expert Pro", layout="wide")
st.header("AI-Powered Resume Optimization System")

# Input Sections
col1, col2 = st.columns(2)
with col1:
    job_desc = st.text_area("📋 Paste Job Description:", height=200, key="jd")
with col2:
    resume_file = st.file_uploader("📤 Upload Resume (PDF):", type=["pdf"])

# New Feature: Q&A Input
user_question = st.text_area("❓ Ask a question about your resume:", key="qa", disabled=not resume_file)

# Action Buttons
actions = st.columns(5)
with actions[0]:
    analyze_btn = st.button("🔍 Analyze Resume Match")
with actions[1]:
    improve_btn = st.button("📈 Improve Skills")
with actions[2]:
    keywords_btn = st.button("🔑 Missing Keywords")
with actions[3]:
    qna_btn = st.button("💬 Q&A Response")
with actions[4]:
    full_review_btn = st.button("📝 Comprehensive Review")

# Refined Prompts
PROMPTS = {
    "analysis": """As a senior ATS analyst, strictly provide:
1. Match percentage (0-100%) based on required skills/experience
2. Top 3 strengths aligning with job requirements
3. Top 3 weaknesses/missing qualifications
4. Urgency level (Low/Medium/High) for application
Format: Bullet points with emojis, under 200 words""",

    "improvement": """As a career coach, provide:
1. 3 technical skills to develop with learning resources
2. 2 soft skills to highlight
3. Resume formatting improvements
4. Project suggestions to bridge gaps
Format: Numbered list with brief explanations""",

    "keywords": """As an ATS keyword optimizer:
1. List up to 10 missing hard skills from JD
2. 5 missing soft skills/terms
3. Suggested keyword integration strategies
Format: Two-column table (Missing Keywords | Suggested Placement)""",

    "qna": """Answer questions about the resume strictly based on its content:
1. Keep responses under 100 words
2. Highlight relevant sections from resume
3. Provide specific examples when possible
4. If unsure, request clarification
Format: Concise paragraph with 🔍 emoji""",

    "review": """As an HR executive, provide:
1. Overall suitability assessment
2. Experience alignment analysis
3. Education/certification evaluation
4. Recommendation for next steps
Format: Professional report structure with headings"""
}


def handle_response(btn_type, prompt_key):
    if not resume_file:
        st.warning("Please upload a resume first")
        return
    try:
        pdf_content = input_pdf_setup(resume_file)
        response = get_gemini_response(
            PROMPTS[prompt_key],
            pdf_content,
            user_input=job_desc,
            question=user_question if prompt_key == "qna" else None
        )

        st.subheader("Analysis Results")

        if prompt_key == "improvement":
            st.write("### Recommended Skill Improvements")
            for line in response.split('\n'):
                if line.strip():
                    st.write(f"- {line.strip()}")
        else:
            st.write(response)

    except Exception as e:
        st.error(f"Error: {str(e)}")

if analyze_btn:
    handle_response("analysis", "analysis")

elif improve_btn:
    handle_response("improvement", "improvement")

elif keywords_btn:
    handle_response("keywords", "keywords")

elif qna_btn and user_question:
    handle_response("qna", "qna")

elif full_review_btn:
    handle_response("review", "review")

elif qna_btn and not user_question:
    st.warning("Please enter a question first")

# Add visual separator
st.markdown("---")
st.caption("💡 Tip: Upload a resume and job description for comprehensive analysis")