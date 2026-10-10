"""Transparent resume matching using TF-IDF and job-description skills."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_SKILLS = [
    "Python",
    "Java",
    "JavaScript",
    "C++",
    "C#",
    "SQL",
    "Machine Learning",
    "Deep Learning",
    "Natural Language Processing",
    "NLP",
    "Statistics",
    "Data Analysis",
    "Data Visualization",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "Keras",
    "Excel",
    "Power BI",
    "Tableau",
    "Git",
    "GitHub",
    "AWS",
    "Azure",
    "Google Cloud",
    "Docker",
    "Linux",
    "Spark",
    "Hadoop",
    "R",
    "Communication",
    "Project Management",
]


def detect_skills(text: str, skills: Sequence[str] = DEFAULT_SKILLS) -> list[str]:
    """Return listed skills found in text, preserving the dictionary order."""
    if not text:
        return []

    found = []
    for skill in skills:
        if not skill.strip():
            continue
        pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in skill.split()) + r"(?!\w)"
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(skill)
    return found


def _similarities(job_description: str, resume_texts: Sequence[str]) -> list[float]:
    """Calculate cosine similarities on a 0-100 scale; empty text scores zero."""
    if not job_description.strip() or not resume_texts:
        return [0.0] * len(resume_texts)

    documents = [job_description, *resume_texts]
    try:
        matrix = TfidfVectorizer().fit_transform(documents)
    except ValueError:
        return [0.0] * len(resume_texts)

    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    return [float(score * 100) for score in scores]


def rank_resumes(
    job_description: str,
    resumes: Mapping[str, str],
    skills: Sequence[str] = DEFAULT_SKILLS,
    similarity_weight: float = 0.6,
    skills_weight: float = 0.4,
) -> list[dict[str, object]]:
    """Score and rank resumes; weights are fractions and must total 1.0."""
    if similarity_weight < 0 or skills_weight < 0:
        raise ValueError("Scoring weights cannot be negative.")
    if abs(similarity_weight + skills_weight - 1.0) > 1e-9:
        raise ValueError("Similarity and skills weights must total 1.0.")

    job_skills = detect_skills(job_description, skills)
    resume_names = list(resumes)
    resume_texts = [resumes[name] or "" for name in resume_names]
    similarity_scores = _similarities(job_description, resume_texts)

    results: list[dict[str, object]] = []
    for name, text, similarity in zip(resume_names, resume_texts, similarity_scores):
        candidate_skills = detect_skills(text, skills)
        matched_skills = [skill for skill in job_skills if skill in candidate_skills]
        missing_skills = [skill for skill in job_skills if skill not in candidate_skills]
        coverage = (len(matched_skills) / len(job_skills) * 100) if job_skills else 0.0
        overall = similarity * similarity_weight + coverage * skills_weight
        results.append(
            {
                "filename": name,
                "similarity_score": similarity,
                "skills_coverage": coverage,
                "overall_score": overall,
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
            }
        )

    results.sort(key=lambda result: (-float(result["overall_score"]), str(result["filename"]).casefold()))
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    return results