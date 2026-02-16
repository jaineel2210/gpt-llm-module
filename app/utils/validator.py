import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.schemas import EvaluationOutput, EvaluationScores, AntiCheatDetection

logger = logging.getLogger(__name__)

# Legacy validation for backward compatibility
REQUIRED_KEYS = [
    "relevance_score",
    "clarity_score", 
    "technical_accuracy",
    "communication_score",
    "overall_score",
    "feedback"
]

# Comprehensive validation for new system
COMPREHENSIVE_REQUIRED_KEYS = [
    "scores",
    "feedback", 
    "anti_cheat",
    "keyword_analysis",
    "response_quality",
    "areas_for_improvement",
    "processing_metadata"
]

SCORE_KEYS = ["technical_accuracy", "concept_clarity", "keyword_coverage", "communication", "final_score", "confidence_score"]
ANTI_CHEAT_KEYS = ["is_copy_paste", "is_ai_generated", "is_too_robotic", "transcript_mismatch", "confidence_level", "risk_factors"]


def validate_llm_output(output: str) -> Optional[Dict[str, Any]]:
    """
    Legacy validation function for backward compatibility.
    Validates basic LLM output structure.
    """
    try:
        data = json.loads(output)
        
        for key in REQUIRED_KEYS:
            if key not in data:
                logger.warning(f"Missing required key in legacy output: {key}")
                return None
                
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in legacy validation: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in legacy validation: {e}")
        return None


def validate_comprehensive_output(output: str) -> Optional[EvaluationOutput]:
    """
    Comprehensive validation for the new evaluation system.
    
    Args:
        output: JSON string from LLM evaluation
        
    Returns:
        Validated EvaluationOutput object or None if validation fails
    """
    try:
        # Parse JSON
        data = json.loads(output)
        
        # Validate top-level structure
        for key in COMPREHENSIVE_REQUIRED_KEYS:
            if key not in data:
                logger.error(f"Missing required key: {key}")
                return None
        
        # Validate scores structure
        scores_data = data.get("scores", {})
        if not _validate_scores_structure(scores_data):
            logger.error("Invalid scores structure")
            return None
            
        # Validate anti-cheat structure
        anti_cheat_data = data.get("anti_cheat", {})
        if not _validate_anti_cheat_structure(anti_cheat_data):
            logger.error("Invalid anti-cheat structure")
            return None
            
        # Create and validate Pydantic objects
        try:
            evaluation_output = EvaluationOutput(**data)
            return evaluation_output
            
        except Exception as e:
            logger.error(f"Pydantic validation failed: {e}")
            return None
            
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in comprehensive validation: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in comprehensive validation: {e}")
        return None


def _validate_scores_structure(scores_data: Dict[str, Any]) -> bool:
    """Validate the scores sub-structure"""
    
    for key in SCORE_KEYS:
        if key not in scores_data:
            logger.warning(f"Missing score key: {key}")
            return False
            
        value = scores_data[key]
        
        # Validate score ranges
        if key == "confidence_score":
            if not isinstance(value, (int, float)) or not (1 <= value <= 10):
                logger.warning(f"Invalid confidence_score: {value} (should be 1-10)")
                return False
        else:
            if not isinstance(value, (int, float)) or not (0 <= value <= 10):
                logger.warning(f"Invalid score {key}: {value} (should be 0-10)")
                return False
                
    return True


def _validate_anti_cheat_structure(anti_cheat_data: Dict[str, Any]) -> bool:
    """Validate the anti-cheat sub-structure"""
    
    for key in ANTI_CHEAT_KEYS:
        if key not in anti_cheat_data:
            logger.warning(f"Missing anti-cheat key: {key}")
            return False
    
    # Validate boolean fields
    boolean_fields = ["is_copy_paste", "is_ai_generated", "is_too_robotic", "transcript_mismatch"]
    for field in boolean_fields:
        if not isinstance(anti_cheat_data[field], bool):
            logger.warning(f"Anti-cheat field {field} should be boolean")
            return False
    
    # Validate confidence level
    confidence = anti_cheat_data["confidence_level"]
    if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
        logger.warning(f"Invalid confidence_level: {confidence} (should be 0-1)")
        return False
        
    # Validate risk factors
    risk_factors = anti_cheat_data["risk_factors"]
    if not isinstance(risk_factors, list):
        logger.warning("risk_factors should be a list")
        return False
        
    return True


def sanitize_llm_output(output: str) -> str:
    """
    Clean and sanitize LLM output before validation.
    Removes common formatting issues that can cause JSON parsing failures.
    """
    try:
        # Remove common markdown formatting
        cleaned = output.strip()
        
        # Remove code block markers
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
            
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
            
        # Remove any leading/trailing whitespace
        cleaned = cleaned.strip()
        
        # Try to fix common JSON issues
        # Remove trailing commas before closing braces/brackets
        import re
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        
        # Ensure proper quote formatting
        cleaned = cleaned.replace("'", '"')
        
        return cleaned
        
    except Exception as e:
        logger.warning(f"Error sanitizing output: {e}")
        return output


def validate_keyword_analysis(keyword_analysis: Dict[str, Any], expected_keywords: List[str]) -> bool:
    """
    Validate that keyword analysis includes all expected keywords.
    
    Args:
        keyword_analysis: Dictionary mapping keywords to boolean coverage
        expected_keywords: List of keywords that should be analyzed
        
    Returns:
        True if all expected keywords are present in analysis
    """
    try:
        for keyword in expected_keywords:
            if keyword not in keyword_analysis:
                logger.warning(f"Missing keyword in analysis: {keyword}")
                return False
                
            if not isinstance(keyword_analysis[keyword], bool):
                logger.warning(f"Keyword analysis for '{keyword}' should be boolean")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Error validating keyword analysis: {e}")
        return False


def create_validation_report(evaluation_output: EvaluationOutput) -> Dict[str, Any]:
    """
    Create a comprehensive validation report for debugging and monitoring.
    
    Args:
        evaluation_output: The evaluation output to analyze
        
    Returns:
        Dictionary containing validation metrics and warnings
    """
    report = {
        "validation_timestamp": datetime.utcnow().isoformat(),
        "overall_valid": True,
        "warnings": [],
        "metrics": {}
    }
    
    try:
        # Check score consistency
        scores = evaluation_output.scores
        calculated_final = (
            scores.technical_accuracy * 0.4 +
            scores.concept_clarity * 0.25 +
            scores.keyword_coverage * 0.2 +
            scores.communication * 0.15
        )
        
        score_difference = abs(calculated_final - scores.final_score)
        if score_difference > 0.1:
            report["warnings"].append(f"Final score calculation mismatch: {score_difference:.2f}")
        
        # Check anti-cheat consistency
        anti_cheat = evaluation_output.anti_cheat
        risk_count = sum([
            anti_cheat.is_copy_paste,
            anti_cheat.is_ai_generated,
            anti_cheat.is_too_robotic,
            anti_cheat.transcript_mismatch
        ])
        
        if risk_count > 0 and anti_cheat.confidence_level < 0.3:
            report["warnings"].append("High risk detected but low confidence level")
        
        # Add metrics
        report["metrics"] = {
            "final_score": scores.final_score,
            "confidence_score": scores.confidence_score,
            "risk_factor_count": len(anti_cheat.risk_factors),
            "cheat_probability": anti_cheat.confidence_level,
            "response_length": len(evaluation_output.feedback.split())
        }
        
        # Overall validation status
        report["overall_valid"] = len(report["warnings"]) == 0
        
    except Exception as e:
        report["warnings"].append(f"Validation report generation failed: {str(e)}")
        report["overall_valid"] = False
    
    return report
