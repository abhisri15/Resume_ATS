from flask import Flask, request, jsonify
import os
import re
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from processor import extract_resume_data
from resume_category_predict import predict_category, extract_text_from_pdf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load Spacy's English model
nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(lemmas)


def calculate_matching_score(resume_data, job_description):
    sections = {
        'skills': ' '.join(resume_data.get('skills', [])),
        'experience': ' '.join([
            f"{exp.get('role', '')} {exp.get('company', '')} {' '.join(exp.get('contributions', []))}"
            for exp in resume_data.get('experience', [])
        ]),
        'education': ' '.join([
            f"{edu.get('degree', '')} {edu.get('university', '')}"
            for edu in resume_data.get('education', [])
        ])
    }

    preprocessed_jd = preprocess_text(job_description)
    preprocessed_sections = {k: preprocess_text(v) for k, v in sections.items()}

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    jd_vector = vectorizer.fit_transform([preprocessed_jd])

    section_weights = {'skills': 0.4, 'experience': 0.5, 'education': 0.1}
    similarity_scores = {}

    for section, text in preprocessed_sections.items():
        section_vector = vectorizer.transform([text])
        similarity = cosine_similarity(jd_vector, section_vector)[0][0]
        similarity_scores[section] = similarity

    weighted_score = sum(similarity_scores[section] * weight
                         for section, weight in section_weights.items())
    return min(max(weighted_score, 0), 1)


@app.route('/', methods=['POST'])
def parse_resume():
    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not (file and file.filename.lower().endswith('.pdf')):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    try:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Process resume
        results = extract_resume_data(filename)
        resume_text = extract_text_from_pdf(filename)
        results["predicted_category"] = predict_category(resume_text)

        os.remove(filename)
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/match', methods=['POST'])
def match_resume():
    try:
        data = request.get_json()

        if not data or 'resume_data' not in data or 'job_description' not in data:
            return jsonify({"error": "Both resume_data and job_description are required"}), 400

        resume_data = data['resume_data']
        job_description = data['job_description']

        if not isinstance(resume_data, dict) or not isinstance(job_description, str):
            return jsonify({"error": "Invalid input format"}), 400

        score = calculate_matching_score(resume_data, job_description)
        resume_data['matching_score'] = round(score, 2)
        resume_data['match_analysis'] = "Strong match" if score > 0.7 else \
            "Moderate match" if score > 0.5 else "Weak match"

        return jsonify(resume_data)

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)