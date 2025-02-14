import docx2txt
from PyPDF2 import PdfReader

# Extracting text from PDF
def pdftotext(m):
    with open(m, 'rb') as pdfFileObj:
        pdf_reader = PdfReader(pdfFileObj)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ''
    return text

if __name__ == '__main__':

    FilePath = 'AI.pdf'
    FilePath.lower().endswith(('.png', '.docx'))
    if FilePath.endswith('.docx'):
        textinput = doctotext(FilePath)
    elif FilePath.endswith('.pdf'):
        textinput = pdftotext(FilePath)
    else:
        print("File not support")

import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

stop = stopwords.words('english')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

import spacy
import en_core_web_sm
from spacy.matcher import Matcher

# load pre-trained model
nlp = en_core_web_sm.load()

# initialize matcher with a vocab
matcher = Matcher(nlp.vocab)


def extract_name(text):
    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)

    pattern = [{"POS": "PROPN"}, {"POS": "PROPN"}]  # Matches two proper nouns (e.g., first & last name)

    matcher.add('NAME', [pattern])  # Correct way to add patterns

    doc = nlp(text)
    matches = matcher(doc)

    for match_id, start, end in matches:
        return doc[start:end].text  # Return the first matched name

    return "Name not found"


print('Name: ', extract_name(textinput))

import re
from nltk.corpus import stopwords

# Grad all general stop words
STOPWORDS = set(stopwords.words('english'))

# Education Degrees
EDUCATION = [
    'BE', 'B.E.', 'B.E', 'BS', 'B.S',
    'ME', 'M.E', 'M.E.', 'M.B.A', 'MBA', 'MS', 'M.S',
    'BTECH', 'B.TECH', 'M.TECH', 'MTECH',
    'SSLC', 'SSC' 'HSC', 'CBSE', 'ICSE', 'X', 'XII'
]


def extract_education(resume_text):
    nlp_text = nlp(resume_text)

    # Sentence Tokenizer
    nlp_text = [sent.text.strip() for sent in nlp_text.sents]

    edu = {}
    # Extract education degree
    for index, text in enumerate(nlp_text):
        for tex in text.split():
            # Replace all special symbols
            tex = re.sub(r'[?|$|.|!|,]', r'', tex)
            if tex.upper() in EDUCATION and tex not in STOPWORDS:
                edu[tex] = text + nlp_text[index + 1]

    # Extract year
    education = []
    for key in edu.keys():
        year = re.search(re.compile(r'(((20|19)(\d{})))'), edu[key])
        if year:
            education.append((key, ''.join(year[0])))
        else:
            education.append(key)
    return education


print('Qualification: ', extract_education(textinput))

import pandas as pd
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp(textinput)
noun_chunks = list(doc.noun_chunks)



def extract_skills(resume_text):
    nlp_text = nlp(resume_text)

    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]
    colnames = ['skill']
    # reading the csv file
    data = pd.read_csv('skill.csv', names=colnames)

    # extract values
    skills = data.skill.tolist()
    print(skills)
    skillset = []

    # check for one-grams (example: python)
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)

    for token in noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)
    return [i.capitalize() for i in set([i.lower() for i in skillset])]


print('Skills', extract_skills(textinput))


def extract_mobile_number(resume_text):
    phone = re.findall(re.compile(
        r'(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?'),
                       resume_text)

    if phone:
        number = ''.join(phone[0])
        if len(number) > 10:
            return number
        else:
            return number


print('Mobile Number: ', extract_mobile_number(textinput))


def extract_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)


print('Mail id: ', extract_email_addresses(textinput))