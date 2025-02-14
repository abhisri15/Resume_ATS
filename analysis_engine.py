import re


class ResumeAnalyzer:
    def __init__(self, data):
        self.data = data

    def get_improvements(self):
        suggestions = []

        # Check metrics in projects
        projects_text = ' '.join(self.data['projects'])
        if not re.search(r'\d+%', projects_text):
            suggestions.append("Add quantitative metrics to project descriptions")

        # Check action verbs
        verbs = ['developed', 'created', 'implemented', 'designed', 'built']
        if not any(verb in projects_text.lower() for verb in verbs):
            suggestions.append("Use more action verbs in experience/projects")

        # Check skills
        if len(self.data['technical_skills']) < 8:
            suggestions.append("Include more detailed technical skills")

        # Check education dates
        education_text = ' '.join(self.data['education'])
        if not re.search(r'(20\d{2}\s*-\s*20\d{2}|present)', education_text):
            suggestions.append("Add timeline/duration for education")

        return suggestions