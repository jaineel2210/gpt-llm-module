# app/main.py
from fastapi import FastAPI, HTTPException
from app.schemas import EvaluationInput, EvaluationOutput
from app.services.llm_evaluator import evaluate_interview_answer
from app.utils.validator import validate_llm_output
from openai import RateLimitError, APIError, AuthenticationError
import json

app = FastAPI()

@app.post("/evaluate")
def evaluate(data: EvaluationInput):
    """
    Evaluate an interview answer based on question, answer, and experience level.
    Returns structured evaluation with scores and feedback.
    """
    try:
        result = evaluate_interview_answer(
            question=data.question,
            answer=data.answer,
            experience_level=data.experience_level
        )
        
        # Validate the LLM output
        validated_result = validate_llm_output(result)
        
        if validated_result:
            return validated_result
        else:
            # If validation fails, return raw result with error indication
            return {
                "error": "Failed to parse LLM response",
                "raw_response": result
            }
    
    except RateLimitError as e:
        raise HTTPException(
            status_code=429,
            detail="OpenAI API quota exceeded. Please check your plan and billing details."
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=401,
            detail="Invalid OpenAI API key. Please check your API key configuration."
        )
    except APIError as e:
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI API error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
