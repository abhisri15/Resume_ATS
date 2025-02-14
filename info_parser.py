import re
import spacy
from collections import defaultdict
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")


class InfoParser:
    def __init__(self, text):
        self.text = text
        self.lines = [line.strip() for line in text.split('\n') if line.strip()]

    def parse(self):
        return {
            **self._extract_personal_info(),
            **self._extract_sections()
        }

    def _extract_personal_info(self):
        return {
            'name': self._extract_name(),
            'email': self._extract_email(),
            'phone': self._extract_phone(),
            'location': self._extract_location()
        }

    def _extract_name(self):
        nlp = spacy.load('en_core_web_sm')
        matcher = Matcher(nlp.vocab)

        pattern = [[{"POS": "PROPN"}, {"POS": "PROPN"}]]  # Matches two proper nouns (e.g., "John Doe")
        matcher.add("NAME", pattern)  # Corrected syntax

        # Check first 3 lines for name pattern
        for line in self.lines[:3]:
            # Regex-based check for capitalized words (backup method)
            if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$', line):
                return line

            doc = nlp(line)
            matches = matcher(doc)

            for match_id, start, end in matches:
                return doc[start:end].text  # Returns the first matched name

        return ' '.join(self.lines[0].split()[:3])  # Fallback: first three words of the first line

    def _extract_email(self):
        email = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', self.text)
        return email.group(0) if email else ''

    def _extract_phone(self):
        phone = re.search(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', self.text)
        return phone.group(0) if phone else ''

    def _extract_location(self):
        cities = ['bangalore', 'delhi', 'hyderabad', 'mumbai', 'chennai',
                  'kolkata', 'pune', 'noida', 'gurgaon', 'bengaluru', 'bhubaneswar']
        pattern = re.compile(rf'\b({"|".join(cities)})\b', re.IGNORECASE)

        for line in self.lines[:5]:
            match = pattern.search(line)
            if match:
                return match.group().title()
        return ''

    def _extract_sections(self):
        section_headers = {
            'education': r'(education|academic background)',
            'experience': r'(experience|work history)',
            'projects': r'(projects|personal projects)',
            'technical_skills': r'(technical skills|skills)',
            'achievements': r'(achievements|awards)',
            'research': r'(research projects|publications)'
        }

        sections = defaultdict(list)
        current_section = None
        section_pattern = re.compile(
            rf'^({"|".join(section_headers.values())})$',
            re.IGNORECASE
        )

        for line in self.lines:
            # Detect section headers
            match = section_pattern.search(line)
            if match:
                current_section = next(
                    k for k, v in section_headers.items()
                    if re.search(v, line, re.IGNORECASE)
                )
                continue

            if current_section:
                self._process_line(line, sections[current_section])

        return {
            'education': self._process_education(sections['education']),
            'experience': self._process_experience(sections['experience']),
            'projects': self._process_projects(sections['projects']),
            'technical_skills': self._process_skills(sections['technical_skills']),
            'achievements': sections['achievements'],
            'research': sections['research']
        }

    def _process_line(self, line, section):
        if not line:
            return

        # Merge bullet points
        if line.startswith(('•', '-', '*')):
            section.append(line[1:].strip())
        elif section and ':' not in line:
            section[-1] += ' ' + line
        else:
            section.append(line)

    def _process_education(self, edu_lines):
        education = []
        current = []
        # Match education entry patterns
        edu_pattern = re.compile(
            r'(?:Bachelor|Master|Ph\.?D|High School|School|College|University|Institute|'
            r'\b(?:B\.?Tech|M\.?Tech|B\.?E|M\.?E|B\.?Sc|M\.?Sc)\b|'
            r'\d{2}th Standard|\d{1,2}(?:st|nd|rd|th) Grade|CGPA|GPA|%)',
            re.IGNORECASE
        )

        for line in edu_lines:
            if edu_pattern.search(line) and current:
                if any(char.isdigit() for char in line[:4]):
                    education.append(' '.join(current))
                    current = []
            current.append(line)

        if current:
            education.append(' '.join(current))

        # Post-process to clean up
        return [re.sub(r'\s+', ' ', edu).strip() for edu in education]

    def _process_experience(self, exp_lines):
        experience = []
        current = []
        # Improved experience pattern with company detection
        exp_pattern = re.compile(
            r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s*-\s*'
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b|'
            r'\b(Present|Current)\b)|'
            r'^[A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*$',
            re.IGNORECASE
        )

        for line in exp_lines:
            if exp_pattern.search(line):
                if current:
                    experience.append(self._format_experience(current))
                    current = []
            current.append(line)

        if current:
            experience.append(self._format_experience(current))
        return experience

    def _format_experience(self, experience):
        formatted = []
        header = []
        bullets = []

        for line in experience:
            if re.match(r'^\s*•', line):
                bullets.append(line.strip())
            elif not header:
                header.append(line)
            else:
                formatted.append(line)

        # Structure the experience entry
        result = []
        if header:
            result.append(' '.join(header))
        if formatted:
            result.extend(formatted)
        if bullets:
            result.append('\n'.join(bullets))

        return '\n'.join(result)

    def _process_projects(self, project_lines):
        projects = []
        current = []
        # Enhanced project header detection
        project_header = re.compile(
            r'^([A-Z][\w\s]+?)\s*[\|•\-]|'
            r'\b(Project|Initiative|Research|Development)\b',
            re.IGNORECASE
        )

        for line in project_lines:
            if project_header.match(line) and current:
                projects.append(self._format_project(current))
                current = []
            current.append(line)

        if current:
            projects.append(self._format_project(current))

        return projects

    def _format_project(self, project):
        # Split into title and description
        title = []
        bullets = []
        description = []

        for line in project:
            if re.match(r'^\s*•', line):
                bullets.append(line.strip())
            elif not title and re.search(r'\b(Tech|Stack|Tools|Used)\b', line):
                title.append(line)
            elif not title:
                title.append(line)
            else:
                description.append(line)

        formatted = []
        if title:
            formatted.append(' '.join(title))
        if description:
            formatted.append(' '.join(description))
        if bullets:
            formatted.append('\n'.join(bullets))

        return '\n'.join(formatted)
    def _process_skills(self, skill_lines):
        skills = []
        for line in skill_lines:
            if ':' in line:
                skills.extend([s.strip() for s in line.split(':')[1].split(',')])
            else:
                skills.extend([s.strip() for s in line.split(',')])
        return [s for s in skills if s]