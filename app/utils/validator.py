import json

REQUIRED_KEYS = [
    "relevance_score",
    "clarity_score",
    "technical_accuracy",
    "communication_score",
    "overall_score",
    "feedback"
]


def validate_llm_output(output: str):
    try:
        data = json.loads(output)

        for key in REQUIRED_KEYS:
            if key not in data:
                return None

        return data

    except Exception:
        return None
