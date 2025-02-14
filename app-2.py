from flask import Flask, render_template, request
from info_parser import InfoParser
from analysis_engine import ResumeAnalyzer

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        resume_text = request.form['resume_text']
        parser = InfoParser(resume_text)
        parsed_data = parser.parse()
        analyzer = ResumeAnalyzer(parsed_data)

        results = {
            **parsed_data,
            'improvements': analyzer.get_improvements()
        }
        return render_template('results.html', results=results)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)