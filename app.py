"""Streamlit interface for transparent, skills-based resume screening."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from matcher import DEFAULT_SKILLS, rank_resumes
from resume_parser import extract_text_from_pdf

st.set_page_config(page_title="AI Resume Screening System", page_icon="📄", layout="wide")
st.title("AI Resume Screening System")
st.caption("Compare resume text with a job description using TF-IDF and an editable skills list.")
st.warning(
    "Screening scores are estimates, not measures of a candidate's potential. "
    "Do not use them as the sole basis for hiring decisions. Review resumes fairly and independently."
)
st.info(
    "This tool does not infer or score age, gender, religion, caste, disability, photographs, "
    "or other sensitive personal attributes."
)

job_description = st.text_area("Job description", height=200, placeholder="Paste the role requirements here...")
uploaded_files = st.file_uploader(
    "Upload one or more PDF resumes", type=["pdf"], accept_multiple_files=True
)

with st.expander("Skill dictionary", expanded=False):
    skill_text = st.text_area(
        "One skill per line",
        value="\n".join(DEFAULT_SKILLS),
        height=240,
        help="Edit this list to reflect the role. Matching is case-insensitive.",
    )

st.subheader("Score weights")
similarity_percent = st.slider("TF-IDF text similarity", min_value=0, max_value=100, value=60, step=5)
skills_percent = 100 - similarity_percent
st.caption(f"Skills coverage: {skills_percent}% (the weights always total 100%).")

if st.button("Screen resumes", type="primary", use_container_width=True):
    if not job_description.strip():
        st.error("Enter a job description before screening resumes.")
    elif not uploaded_files:
        st.error("Upload at least one PDF resume before screening.")
    else:
        resume_texts: dict[str, str] = {}
        file_errors: list[str] = []
        for uploaded_file in uploaded_files:
            try:
                resume_texts[uploaded_file.name] = extract_text_from_pdf(uploaded_file.getvalue())
            except ValueError as error:
                file_errors.append(f"{uploaded_file.name}: {error}")

        for file_error in file_errors:
            st.warning(file_error)

        if not resume_texts:
            st.error("No readable resumes were found. Upload PDFs that contain selectable text.")
        else:
            skills = [skill.strip() for skill in skill_text.splitlines() if skill.strip()]
            results = rank_resumes(
                job_description,
                resume_texts,
                skills=skills,
                similarity_weight=similarity_percent / 100,
                skills_weight=skills_percent / 100,
            )
            job_skills = [skill for skill in skills if skill in results[0]["matched_skills"] or skill in results[0]["missing_skills"]]
            if not job_skills:
                st.warning("No skills from the current dictionary were found in the job description; skills coverage is 0%.")

            table_rows = [
                {
                    "Rank": result["rank"],
                    "Candidate file": result["filename"],
                    "Overall score (%)": round(float(result["overall_score"]), 2),
                    "TF-IDF similarity (%)": round(float(result["similarity_score"]), 2),
                    "Skills coverage (%)": round(float(result["skills_coverage"]), 2),
                    "Matched skills": ", ".join(result["matched_skills"]),
                    "Missing skills": ", ".join(result["missing_skills"]),
                }
                for result in results
            ]
            results_frame = pd.DataFrame(table_rows)
            st.subheader("Ranked results")
            st.dataframe(results_frame, use_container_width=True, hide_index=True)
            csv_data = results_frame.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download results as CSV",
                data=csv_data,
                file_name="resume_screening_results.csv",
                mime="text/csv",
            )

            for result in results:
                with st.expander(f"#{result['rank']} {result['filename']} | {float(result['overall_score']):.1f}%"):
                    st.write("**Matched skills:**", ", ".join(result["matched_skills"]) or "None")
                    st.write("**Missing skills:**", ", ".join(result["missing_skills"]) or "None")