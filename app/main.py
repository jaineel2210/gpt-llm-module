# app/main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import logging
import json

from app.schemas import (
    EvaluationInput, EvaluationOutput, ScoringWeights, RiskEngineOutput,
    ExperienceLevel, QuestionType
)
from app.services.llm_evaluator import LLMEvaluator, evaluate_interview_answer
from app.utils.validator import (
    validate_llm_output, validate_comprehensive_output, 
    sanitize_llm_output, create_validation_report
)
from openai import RateLimitError, APIError, AuthenticationError

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced LLM Interview Evaluation System",
    description="Comprehensive AI-powered interview evaluation with anti-cheating detection",
    version="2.0.0"
)

# Add CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global evaluator instance
evaluator = LLMEvaluator()

@app.get("/")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "LLM Interview Evaluation System",
        "version": "2.0.0",
        "features": [
            "Comprehensive scoring", 
            "Anti-cheating detection", 
            "Risk assessment",
            "Integration ready"
        ]
    }

@app.post("/evaluate/comprehensive", response_model=EvaluationOutput)
def evaluate_comprehensive(evaluation_input: EvaluationInput):
    """
    Comprehensive interview evaluation with advanced features:
    - Weighted scoring rubric (Technical 40%, Clarity 25%, Keywords 20%, Communication 15%)
    - Anti-cheating detection (copy-paste, AI-generated, robotic responses)
    - Detailed feedback and improvement suggestions
    - Risk assessment for integration with monitoring systems
    """
    try:
        logger.info(f"Starting comprehensive evaluation for {evaluation_input.question_type} question")
        
        # Perform comprehensive evaluation
        result = evaluator.evaluate_answer(evaluation_input)
        
        logger.info(f"Evaluation completed. Final score: {result.scores.final_score}")
        return result
        
    except RateLimitError:
        logger.error("OpenAI API rate limit exceeded")
        raise HTTPException(
            status_code=429,
            detail="API rate limit exceeded. Please try again later."
        )
    except AuthenticationError:
        logger.error("OpenAI API authentication failed")
        raise HTTPException(
            status_code=401,
            detail="Authentication failed. Please check API configuration."
        )
    except APIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"External API error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Comprehensive evaluation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )

@app.post("/evaluate/risk-engine", response_model=RiskEngineOutput)
def evaluate_for_risk_engine(evaluation_input: EvaluationInput):
    """
    Evaluation endpoint specifically designed for integration with risk assessment engine.
    
    Returns essential risk metrics:
    - LLM score (0-10)
    - Risk flag (boolean)
    - Confidence level (low/medium/high)
    - Cheating probability (0-1)
    """
    try:
        logger.info("Starting evaluation for risk engine integration")
        
        # Perform comprehensive evaluation
        evaluation_result = evaluator.evaluate_answer(evaluation_input)
        
        # Convert to risk engine format
        risk_output = evaluator.generate_risk_engine_output(evaluation_result)
        
        logger.info(f"Risk evaluation completed. Score: {risk_output.llm_score}, Risk: {risk_output.risk_flag}")
        return risk_output
        
    except Exception as e:
        logger.error(f"Risk engine evaluation failed: {str(e)}")
        # Return safe fallback for risk engine
        return RiskEngineOutput(
            llm_score=5.0,
            risk_flag=True,  # Err on side of caution
            confidence_level="low",
            cheat_probability=0.5,
            quality_metrics={
                "technical_accuracy": 5.0,
                "concept_clarity": 5.0,
                "keyword_coverage": 5.0,
                "communication": 5.0
            },
            evaluation_timestamp="",
            metadata={"error": str(e), "fallback_used": True}
        )

@app.post("/evaluate")
def evaluate_legacy(data: dict):
    """
    Legacy evaluation endpoint for backward compatibility.
    Supports existing integrations while providing enhanced functionality.
    """
    try:
        # Extract legacy format data
        question = data.get("question", "")
        answer = data.get("answer", "")
        experience_level = data.get("experience_level", "intermediate")
        
        if not question or not answer:
            raise HTTPException(
                status_code=400,
                detail="Question and answer are required"
            )
        
        # Use legacy function that returns old format
        result = evaluate_interview_answer(question, answer, experience_level)
        
        # Sanitize and validate
        sanitized_result = sanitize_llm_output(result)
        validated_result = validate_llm_output(sanitized_result)
        
        if validated_result:
            return validated_result
        else:
            logger.warning("Legacy validation failed, returning raw response")
            return {
                "error": "Failed to parse LLM response",
                "raw_response": result
            }
            
    except HTTPException:
        raise
    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="API quota exceeded. Please check your plan and billing details."
        )
    except AuthenticationError:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Please check your configuration."
        )
    except APIError as e:
        raise HTTPException(
            status_code=500,
            detail=f"API error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Legacy evaluation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/evaluate/weights")
def get_scoring_weights():
    """Get current scoring weights configuration"""
    return {
        "weights": {
            "technical_accuracy": evaluator.weights.technical_accuracy,
            "concept_clarity": evaluator.weights.concept_clarity,
            "keyword_coverage": evaluator.weights.keyword_coverage,
            "communication": evaluator.weights.communication
        },
        "description": {
            "technical_accuracy": "Factual correctness and understanding (40%)",
            "concept_clarity": "Clear explanation and logical flow (25%)",
            "keyword_coverage": "Coverage of expected terms/concepts (20%)",
            "communication": "Clarity and professional expression (15%)"
        }
    }

@app.post("/evaluate/validate")
def validate_response(response_data: dict):
    """
    Validation endpoint for testing and debugging LLM responses.
    Helps developers understand why certain responses might fail validation.
    """
    try:
        response_str = json.dumps(response_data) if isinstance(response_data, dict) else response_data
        
        # Try comprehensive validation first
        sanitized = sanitize_llm_output(response_str)
        validated = validate_comprehensive_output(sanitized)
        
        if validated:
            validation_report = create_validation_report(validated)
            return {
                "status": "valid",
                "validation_report": validation_report,
                "sanitized_input": sanitized
            }
        else:
            # Try legacy validation
            legacy_validated = validate_llm_output(sanitized)
            if legacy_validated:
                return {
                    "status": "legacy_valid",
                    "message": "Valid for legacy format but not comprehensive format",
                    "data": legacy_validated
                }
            else:
                return {
                    "status": "invalid",
                    "message": "Failed both comprehensive and legacy validation",
                    "sanitized_input": sanitized
                }
                
    except Exception as e:
        return {
            "status": "error",
            "message": f"Validation error: {str(e)}",
            "input_received": str(response_data)[:500]  # Truncate for safety
        }

@app.get("/evaluate/config")
def get_system_config():
    """Get current system configuration and capabilities"""
    return {
        "model_config": {
            "model_name": "gpt-4o-mini",
            "temperature": 0.2,
            "max_tokens": 1500
        },
        "scoring_rubric": {
            "scale": "0-10",
            "weights": get_scoring_weights()["weights"]
        },
        "anti_cheat_features": [
            "Copy-paste detection",
            "AI-generated content detection", 
            "Robotic response detection",
            "Transcript mismatch analysis"
        ],
        "supported_experience_levels": ["fresher", "intermediate", "advanced"],
        "supported_question_types": ["technical", "behavioral", "system_design", "coding", "conceptual"],
        "integration_endpoints": ["/evaluate/comprehensive", "/evaluate/risk-engine"],
        "legacy_support": True
    }
