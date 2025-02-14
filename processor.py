import os
import fitz
import base64
import google.generativeai as genai
from dotenv import load_dotenv
import json

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

RESUME_PROMPT = """Extract the following details from the resume and return in JSON format:

1. Personal Information:
   - Name
   - Email
   - Phone
   - Location
   - Links (LinkedIn, GitHub, etc.)

2. Education (Include university name, degree, CGPA/percentage, and location)

3. Experience (Include company name, duration, role, and key contributions)

4. Projects (Include name, tech stack, and details of the project)

5. Technical Skills (Languages, Frameworks, Tools and all skills mentioned in resume)

6. Achievements (Certifications, Awards, Hackathon Wins, Research Paper Presentations)

7. Keep "Improvement Suggestions" blank for now.

Ensure the output is structured correctly in JSON format.

Return in this JSON-like format (NO MARKDOWN):
{
  "personal_info": {
    "name": "...",
    "email": "...",
    "phone": "...",
    "location": "...",
    "links": [...]
  },
  "education": ["...", "..."],
  "experience": ["...", "..."],
  "projects": ["...", "..."],
  "skills": ["...", "..."],
  "achievements": ["...", "..."]
}"""


def extract_resume_data(pdf_path):
    try:
        text = extract_text_from_pdf(pdf_path)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([RESUME_PROMPT, text])

        return parse_llm_response(response.text)
    except Exception as e:
        raise RuntimeError(f"LLM processing failed: {str(e)}")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def parse_llm_response(response):
    try:
        # Remove Markdown-style triple backticks and leading/trailing whitespace
        response = response.strip().strip('```json').strip()

        # Parse the JSON
        parsed = json.loads(response)

        # Ensure all fields exist
        personal_info = parsed.get("personal_info", {})
        education = parsed.get("education", [])
        experience = parsed.get("experience", [])
        projects = parsed.get("projects", [])
        skills = parsed.get("skills", [])  # Ensure skills are extracted
        achievements = parsed.get("achievements", [])

        return {
            "name": personal_info.get("name", ""),
            "email": personal_info.get("email", ""),
            "phone": personal_info.get("phone", ""),
            "location": personal_info.get("location", ""),
            "education": education,
            "experience": experience,
            "projects": projects,
            "skills": skills,  # Include skills in final output
            "achievements": achievements,
            "improvement_suggestions": ""  # Keep blank for now
        }

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response: {str(e)}. Response: {response}")
