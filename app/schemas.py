from pydantic import BaseModel

class EvaluationInput(BaseModel):
    question: str
    answer: str
    experience_level: str  # fresher / intermediate / advanced


class EvaluationOutput(BaseModel):
    relevance_score: float
    clarity_score: float
    technical_accuracy: float
    communication_score: float
    overall_score: float
    feedback: str
