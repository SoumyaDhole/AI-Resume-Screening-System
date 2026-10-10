from matcher import detect_skills, rank_resumes


def test_detects_skills_case_insensitively_and_as_phrases():
    detected = detect_skills("Built a MACHINE LEARNING pipeline in Python with scikit-learn.")

    assert detected == ["Python", "Machine Learning", "Scikit-learn"]


def test_empty_text_has_zero_scores_and_no_detected_skills():
    assert detect_skills("") == []

    results = rank_resumes("Python and SQL experience", {"empty.pdf": ""})

    assert results[0]["similarity_score"] == 0
    assert results[0]["skills_coverage"] == 0
    assert results[0]["overall_score"] == 0
    assert results[0]["matched_skills"] == []


def test_ranks_stronger_job_match_first():
    results = rank_resumes(
        "Python developer with SQL and machine learning experience",
        {
            "unrelated.pdf": "Experienced baker and pastry chef",
            "relevant.pdf": "Python developer. SQL data analysis and machine learning projects.",
        },
    )

    assert [result["rank"] for result in results] == [1, 2]
    assert results[0]["filename"] == "relevant.pdf"
    assert results[0]["overall_score"] > results[1]["overall_score"]
    assert results[0]["matched_skills"] == ["Python", "SQL", "Machine Learning"]


def test_weights_must_total_one():
    try:
        rank_resumes("Python", {"resume.pdf": "Python"}, similarity_weight=0.5, skills_weight=0.4)
    except ValueError as error:
        assert "total 1.0" in str(error)
    else:
        raise AssertionError("Expected invalid scoring weights to be rejected")