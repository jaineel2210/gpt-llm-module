# app/services/llm_evaluator.py

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, AuthenticationError

# Load environment variables from .env
load_dotenv()

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_MOCK_MODE = os.getenv("USE_MOCK_MODE", "false").lower() == "true"

if not OPENAI_API_KEY and not USE_MOCK_MODE:
    raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env or USE_MOCK_MODE=true")

# Create OpenAI client only if not in mock mode
client = None
if OPENAI_API_KEY and not USE_MOCK_MODE:
    client = OpenAI(api_key=OPENAI_API_KEY)

# You can set your model here
MODEL_NAME = "gpt-4o-mini"  # Change if you want another model

# Load evaluation prompt template
PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "evaluation_prompt.txt"
with open(PROMPT_FILE, "r") as f:
    EVALUATION_PROMPT_TEMPLATE = f.read()

def evaluate_interview_answer(question: str, answer: str, experience_level: str) -> str:
    """
    Evaluate an interview answer using OpenAI Chat API or mock mode.
    
    Args:
        question: The interview question asked
        answer: The candidate's answer
        experience_level: The candidate's experience level (fresher/intermediate/advanced)
    
    Returns:
        JSON string with evaluation scores and feedback
    """
    # Use mock mode if enabled or if API call fails
    if USE_MOCK_MODE or not client:
        return _generate_mock_evaluation(question, answer, experience_level)
    
    try:
        # Format the prompt with the provided data
        formatted_prompt = EVALUATION_PROMPT_TEMPLATE.format(
            question=question,
            answer=answer,
            experience_level=experience_level
        )
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an AI interview evaluator. You must respond with valid JSON only containing the scores and feedback as specified."},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )

        # Extract model reply
        return response.choices[0].message.content
    
    except (RateLimitError, APIError, AuthenticationError):
        # Fall back to mock mode if API fails
        return _generate_mock_evaluation(question, answer, experience_level)

def _generate_mock_evaluation(question: str, answer: str, experience_level: str) -> str:
    """
    Generate a mock evaluation response for testing purposes.
    """
    # Calculate basic scores based on answer length and content
    answer_length = len(answer.split())
    
    # Simple heuristic scoring
    relevance = min(9.0, 6.0 + (answer_length / 20))
    clarity = min(9.0, 6.5 + (answer_length / 25))
    technical = min(9.0, 7.0 + (answer_length / 30))
    communication = min(9.0, 6.5 + (answer_length / 25))
    overall = round((relevance + clarity + technical + communication) / 4, 1)
    
    mock_response = {
        "relevance_score": round(relevance, 1),
        "clarity_score": round(clarity, 1),
        "technical_accuracy": round(technical, 1),
        "communication_score": round(communication, 1),
        "overall_score": overall,
        "feedback": f"Mock evaluation for {experience_level} level: The answer demonstrates understanding of {question[:50]}... The response is clear and shows technical knowledge. Consider providing more specific examples to strengthen the answer."
    }
    
    return json.dumps(mock_response)

def evaluate_answer(prompt: str) -> str:
    """
    Legacy function: Evaluate a given prompt using OpenAI Chat API.
    Ensures system instructs model to respond in valid JSON only.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You must respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300
    )

    # Extract model reply
    return response.choices[0].message.content
