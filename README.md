# AI Resume Screening System

A small local Streamlit application that compares PDF resume text with a recruiter-provided job description. It uses TF-IDF cosine similarity and a transparent coverage score over skills listed in an editable dictionary. It does not use a trained deep-learning model or an external generative AI service.

## Features

- Paste a job description and upload multiple PDF resumes.
- Extract selectable text from PDFs; unreadable, scanned, and empty files are reported individually.
- Edit the skills dictionary before screening.
- View TF-IDF similarity, skills coverage, combined score, matched skills, and missing skills.
- Adjust the relative weighting of text similarity and skills coverage; the two weights always total 100%.
- Rank candidates and download the result table as CSV.
- The tool does not infer or penalize candidates based on age, gender, religion, caste, disability, photographs, or other sensitive personal attributes.

## Setup

From the project directory in PowerShell, create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `.venv\Scripts\python.exe` in place of `python` in the commands below.

## Run

```powershell
python -m streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) to open in a browser.

## Tests

```powershell
python -m pytest
```

## Methodology

1. PyMuPDF extracts selectable text from each uploaded PDF. Files with no readable text are skipped with a per-file warning.
2. Scikit-learn converts the job description and resume text into TF-IDF vectors and calculates cosine similarity. The value is displayed as a percentage from 0 to 100.
3. The app finds dictionary-listed skills in the job description and each resume using case-insensitive whole-phrase matching. Skills coverage is `matched job-description skills / detected job-description skills * 100`. If no dictionary skill appears in the job description, coverage is 0% and the app warns the recruiter.
4. Overall score is `similarity * similarity weight + skills coverage * skills weight`. Weights are chosen in the UI and sum to 100%.
5. Candidates are sorted by overall score descending. Scores are estimates and can reflect wording, formatting, and dictionary limitations; they are not a hiring recommendation.

## Limitations and responsible use

The application performs text comparison, not a validated assessment of qualifications or job performance. It may miss synonyms, context, skills expressed in unusual formats, and information contained only in images or scanned pages. Review source resumes and use structured, job-related human evaluation. Never use the score as the sole basis for hiring, and do not use sensitive personal attributes in screening decisions.

## Sample testing

See [sample_data/README.md](sample_data/README.md) for instructions to create fictional local test resumes without committing personal data. Tests exercise matching and ranking without requiring any actual candidate data or PDF files.