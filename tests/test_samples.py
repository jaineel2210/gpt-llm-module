from app.utils.validator import validate_llm_output

def test_validator():
    sample_output = """
    {
        "relevance_score": 8,
        "clarity_score": 7,
        "technical_accuracy": 8,
        "communication_score": 7,
        "overall_score": 7.5,
        "feedback": "Good conceptual understanding."
    }
    """

    result = validate_llm_output(sample_output)
    assert result is not None
