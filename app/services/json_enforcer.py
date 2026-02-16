# app/services/json_enforcer.py

import json
import re
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
import asyncio

from app.config import LLM_CONFIG, ERROR_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class JSONValidationResult:
    """Result of JSON validation attempt"""
    is_valid: bool
    parsed_data: Optional[Dict[str, Any]]
    error_type: Optional[str]
    error_message: Optional[str]
    fix_applied: Optional[str]
    attempt_number: int

@dataclass
class JSONEnforcementReport:
    """Comprehensive report of JSON enforcement process"""
    original_response: str
    final_response: Optional[Dict[str, Any]]
    total_attempts: int
    validation_attempts: List[JSONValidationResult]
    fixes_applied: List[str]
    success: bool
    final_error: Optional[str]
    processing_time_ms: float
    
class AdvancedJSONEnforcer:
    """Advanced JSON Output Enforcement with Auto-Retry and Format Fixing"""
    
    def __init__(self, max_attempts: int = 5, enable_auto_fix: bool = True):
        self.max_attempts = max_attempts
        self.enable_auto_fix = enable_auto_fix
        self.fix_patterns = self._initialize_fix_patterns()
        
    def _initialize_fix_patterns(self) -> List[Dict[str, Any]]:
        """Initialize common JSON fix patterns"""
        
        return [
            {
                "name": "Remove Markdown Code Blocks",
                "pattern": r"```(?:json)?\s*(.*?)\s*```",
                "replacement": r"\1",
                "flags": re.DOTALL | re.IGNORECASE
            },
            {
                "name": "Remove Leading/Trailing Text",
                "pattern": r"^[^{]*({.*})[^}]*$", 
                "replacement": r"\1",
                "flags": re.DOTALL
            },
            {
                "name": "Fix Trailing Commas",
                "pattern": r",(\s*[}\]])",
                "replacement": r"\1",
                "flags": re.MULTILINE
            },
            {
                "name": "Fix Single Quotes to Double Quotes",
                "pattern": r"'([^']*)'(\s*:)",
                "replacement": r'"\1"\2',
                "flags": re.MULTILINE
            },
            {
                "name": "Fix Unquoted Keys",
                "pattern": r'(\w+)(\s*:)',
                "replacement": r'"\1"\2',
                "flags": re.MULTILINE
            },
            {
                "name": "Fix Boolean Values",
                "pattern": r'\b(True|False|None)\b',
                "replacement": lambda m: {"True": "true", "False": "false", "None": "null"}[m.group(1)],
                "flags": re.IGNORECASE
            },
            {
                "name": "Remove Multiple Consecutive Commas",
                "pattern": r',(\s*,)+',
                "replacement": r',',
                "flags": re.MULTILINE
            },
            {
                "name": "Fix Missing Commas Between Objects",
                "pattern": r'}\s*{',
                "replacement": '}, {',
                "flags": re.MULTILINE
            },
            {
                "name": "Fix Missing Quotes Around String Values",
                "pattern": r':\s*([a-zA-Z][a-zA-Z0-9_\s]*[a-zA-Z0-9])(\s*[,}])',
                "replacement": r': "\1"\2',
                "flags": re.MULTILINE
            }
        ]
    
    def enforce_json_output(self, llm_response: str, 
                          expected_schema: Optional[Dict[str, Any]] = None) -> JSONEnforcementReport:
        """
        Enforce valid JSON output with multiple validation and fixing attempts
        
        Args:
            llm_response: Raw response from LLM
            expected_schema: Optional schema for validation
            
        Returns:
            JSONEnforcementReport with enforcement results
        """
        start_time = time.time()
        
        validation_attempts = []
        fixes_applied = []
        current_response = llm_response
        
        logger.info("Starting JSON enforcement process")
        
        for attempt in range(1, self.max_attempts + 1):
            logger.debug(f"JSON validation attempt {attempt}")
            
            # First, try to validate as-is
            validation_result = self._validate_json(current_response, attempt, expected_schema)
            validation_attempts.append(validation_result)
            
            if validation_result.is_valid:
                processing_time = (time.time() - start_time) * 1000
                logger.info(f"JSON validation successful on attempt {attempt}")
                
                return JSONEnforcementReport(
                    original_response=llm_response,
                    final_response=validation_result.parsed_data,
                    total_attempts=attempt,
                    validation_attempts=validation_attempts,
                    fixes_applied=fixes_applied,
                    success=True,
                    final_error=None,
                    processing_time_ms=processing_time
                )
            
            # If validation failed and we have attempts left, try to fix
            if attempt < self.max_attempts and self.enable_auto_fix:
                fixed_response, fix_applied = self._apply_json_fixes(current_response)
                if fix_applied:
                    current_response = fixed_response
                    fixes_applied.append(fix_applied)
                    logger.debug(f"Applied fix: {fix_applied}")
                else:
                    # No fix could be applied, try template reconstruction
                    template_fix = self._try_template_reconstruction(current_response, expected_schema)
                    if template_fix:
                        current_response = template_fix
                        fixes_applied.append("Template Reconstruction")
                        logger.debug("Applied template reconstruction fix")
        
        # All attempts failed
        processing_time = (time.time() - start_time) * 1000
        final_error = validation_attempts[-1].error_message if validation_attempts else "Unknown error"
        
        logger.error(f"JSON enforcement failed after {self.max_attempts} attempts: {final_error}")
        
        return JSONEnforcementReport(
            original_response=llm_response,
            final_response=None,
            total_attempts=self.max_attempts,
            validation_attempts=validation_attempts,
            fixes_applied=fixes_applied,
            success=False,
            final_error=final_error,
            processing_time_ms=processing_time
        )
    
    def _validate_json(self, response: str, attempt_number: int,
                      expected_schema: Optional[Dict[str, Any]] = None) -> JSONValidationResult:
        """Validate JSON response and return detailed result"""
        
        try:
            # Remove leading/trailing whitespace
            cleaned_response = response.strip()
            
            # Parse JSON
            parsed_data = json.loads(cleaned_response)
            
            # Validate schema if provided
            if expected_schema:
                schema_validation = self._validate_schema(parsed_data, expected_schema)
                if not schema_validation["valid"]:
                    return JSONValidationResult(
                        is_valid=False,
                        parsed_data=None,
                        error_type="SCHEMA_VIOLATION",
                        error_message=f"Schema validation failed: {schema_validation['error']}",
                        fix_applied=None,
                        attempt_number=attempt_number
                    )
            
            # Additional LLM-specific validation
            llm_validation = self._validate_llm_response_structure(parsed_data)
            if not llm_validation["valid"]:
                return JSONValidationResult(
                    is_valid=False,
                    parsed_data=None,
                    error_type="LLM_STRUCTURE_ERROR",
                    error_message=llm_validation["error"],
                    fix_applied=None,
                    attempt_number=attempt_number
                )
            
            return JSONValidationResult(
                is_valid=True,
                parsed_data=parsed_data,
                error_type=None,
                error_message=None,
                fix_applied=None,
                attempt_number=attempt_number
            )
            
        except json.JSONDecodeError as e:
            return JSONValidationResult(
                is_valid=False,
                parsed_data=None,
                error_type="JSON_DECODE_ERROR",
                error_message=str(e),
                fix_applied=None,
                attempt_number=attempt_number
            )
        except Exception as e:
            return JSONValidationResult(
                is_valid=False,
                parsed_data=None,
                error_type="UNKNOWN_ERROR",
                error_message=str(e),
                fix_applied=None,
                attempt_number=attempt_number
            )
    
    def _validate_schema(self, data: Dict[str, Any], 
                        expected_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against expected schema"""
        
        try:
            # Check required top-level keys
            required_keys = expected_schema.get("required", [])
            for key in required_keys:
                if key not in data:
                    return {"valid": False, "error": f"Missing required key: {key}"}
            
            # Check data types for specific fields
            type_checks = expected_schema.get("types", {})
            for key, expected_type in type_checks.items():
                if key in data and not isinstance(data[key], expected_type):
                    return {"valid": False, "error": f"Invalid type for {key}: expected {expected_type.__name__}"}
            
            # Check value ranges for scores
            score_fields = expected_schema.get("score_fields", [])
            for field in score_fields:
                if field in data:
                    value = data[field]
                    if isinstance(value, (int, float)) and not (0 <= value <= 10):
                        return {"valid": False, "error": f"Score {field} out of range: {value}"}
            
            return {"valid": True, "error": None}
            
        except Exception as e:
            return {"valid": False, "error": f"Schema validation error: {str(e)}"}
    
    def _validate_llm_response_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate LLM-specific response structure"""
        
        try:
            # Check for scores object
            if "scores" in data:
                scores = data["scores"]
                required_score_fields = ["technical_accuracy", "concept_clarity", "keyword_coverage", "communication", "final_score"]
                for field in required_score_fields:
                    if field not in scores:
                        return {"valid": False, "error": f"Missing score field: {field}"}
                    
                    value = scores[field]
                    if not isinstance(value, (int, float)):
                        return {"valid": False, "error": f"Score {field} must be numeric"}
            
            # Check for anti_cheat object
            if "anti_cheat" in data:
                anti_cheat = data["anti_cheat"]
                required_bool_fields = ["is_copy_paste", "is_ai_generated", "is_too_robotic", "transcript_mismatch"]
                for field in required_bool_fields:
                    if field not in anti_cheat:
                        return {"valid": False, "error": f"Missing anti-cheat field: {field}"}
                    
                    if not isinstance(anti_cheat[field], bool):
                        return {"valid": False, "error": f"Anti-cheat field {field} must be boolean"}
            
            # Check for keyword_analysis object
            if "keyword_analysis" in data:
                keyword_analysis = data["keyword_analysis"]
                if not isinstance(keyword_analysis, dict):
                    return {"valid": False, "error": "keyword_analysis must be an object"}
            
            return {"valid": True, "error": None}
            
        except Exception as e:
            return {"valid": False, "error": f"LLM structure validation error: {str(e)}"}
    
    def _apply_json_fixes(self, response: str) -> Tuple[str, Optional[str]]:
        """Apply JSON fixes and return fixed version with description"""
        
        for fix_pattern in self.fix_patterns:
            try:
                if callable(fix_pattern["replacement"]):
                    # Handle function replacements (like boolean fixes)
                    fixed_response = re.sub(
                        fix_pattern["pattern"],
                        fix_pattern["replacement"],
                        response,
                        flags=fix_pattern["flags"]
                    )
                else:
                    # Handle string replacements
                    fixed_response = re.sub(
                        fix_pattern["pattern"],
                        fix_pattern["replacement"],
                        response,
                        flags=fix_pattern["flags"]
                    )
                
                # Check if fix was actually applied
                if fixed_response != response:
                    logger.debug(f"Applied fix: {fix_pattern['name']}")
                    return fixed_response, fix_pattern["name"]
                    
            except Exception as e:
                logger.warning(f"Fix pattern '{fix_pattern['name']}' failed: {str(e)}")
                continue
        
        return response, None
    
    def _try_template_reconstruction(self, response: str, 
                                   expected_schema: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Attempt to reconstruct valid JSON from partial/malformed response"""
        
        try:
            # Extract any JSON-like content
            json_matches = re.findall(r'{[^{}]*}', response, re.DOTALL)
            if not json_matches:
                return None
            
            # Try to build a basic template structure
            template = {
                "scores": {
                    "technical_accuracy": 5.0,
                    "concept_clarity": 5.0,
                    "keyword_coverage": 5.0,
                    "communication": 5.0,
                    "final_score": 5.0,
                    "confidence_score": 5
                },
                "feedback": "Unable to parse original evaluation. Using template response.",
                "anti_cheat": {
                    "is_copy_paste": False,
                    "is_ai_generated": False,
                    "is_too_robotic": False,
                    "transcript_mismatch": False,
                    "confidence_level": 0.1,
                    "risk_factors": ["Template reconstruction used"]
                },
                "keyword_analysis": {},
                "response_quality": "unknown",
                "areas_for_improvement": ["Original response parsing failed"],
                "processing_metadata": {
                    "template_reconstruction": True,
                    "original_response_length": len(response)
                }
            }
            
            # Try to extract some values from the original response
            score_patterns = [
                (r'"technical_accuracy":\s*([\d.]+)', "technical_accuracy"),
                (r'"concept_clarity":\s*([\d.]+)', "concept_clarity"),
                (r'"keyword_coverage":\s*([\d.]+)', "keyword_coverage"),
                (r'"communication":\s*([\d.]+)', "communication"),
                (r'"final_score":\s*([\d.]+)', "final_score")
            ]
            
            for pattern, field in score_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        if 0 <= value <= 10:
                            template["scores"][field] = value
                    except ValueError:
                        pass
            
            # Recalculate final score
            template["scores"]["final_score"] = (
                template["scores"]["technical_accuracy"] * 0.4 +
                template["scores"]["concept_clarity"] * 0.25 +
                template["scores"]["keyword_coverage"] * 0.2 +
                template["scores"]["communication"] * 0.15
            )
            
            return json.dumps(template)
            
        except Exception as e:
            logger.error(f"Template reconstruction failed: {str(e)}")
            return None
    
    def create_llm_schema(self) -> Dict[str, Any]:
        """Create expected schema for LLM evaluation responses"""
        
        return {
            "required": ["scores", "feedback", "anti_cheat", "keyword_analysis", 
                        "response_quality", "areas_for_improvement", "processing_metadata"],
            "types": {
                "scores": dict,
                "feedback": str,
                "anti_cheat": dict,
                "keyword_analysis": dict,
                "response_quality": str,
                "areas_for_improvement": list,
                "processing_metadata": dict
            },
            "score_fields": ["technical_accuracy", "concept_clarity", "keyword_coverage", 
                           "communication", "final_score"]
        }
    
    def generate_enforcement_report(self, enforcement_result: JSONEnforcementReport) -> str:
        """Generate a human-readable enforcement report"""
        
        report_lines = [
            "🔧 JSON ENFORCEMENT REPORT",
            "=" * 40,
            f"Success: {'✅' if enforcement_result.success else '❌'}",
            f"Total Attempts: {enforcement_result.total_attempts}",
            f"Processing Time: {enforcement_result.processing_time_ms:.1f}ms",
            ""
        ]
        
        if enforcement_result.fixes_applied:
            report_lines.append("Fixes Applied:")
            for i, fix in enumerate(enforcement_result.fixes_applied, 1):
                report_lines.append(f"  {i}. {fix}")
            report_lines.append("")
        
        if enforcement_result.success:
            report_lines.append("✅ Final validation successful")
        else:
            report_lines.append(f"❌ Final error: {enforcement_result.final_error}")
        
        report_lines.extend([
            "",
            "Validation Attempts:"
        ])
        
        for attempt in enforcement_result.validation_attempts:
            status = "✅" if attempt.is_valid else "❌"
            report_lines.append(f"  Attempt {attempt.attempt_number}: {status}")
            if not attempt.is_valid:
                report_lines.append(f"    Error: {attempt.error_type} - {attempt.error_message}")
        
        return "\n".join(report_lines)


# Enhanced validator that uses the JSON enforcer
class EnhancedJSONValidator:
    """Enhanced JSON validator with automatic fixing"""
    
    def __init__(self):
        self.enforcer = AdvancedJSONEnforcer(max_attempts=5, enable_auto_fix=True)
        self.schema = self.enforcer.create_llm_schema()
    
    def validate_and_fix(self, llm_response: str) -> Tuple[Optional[Dict[str, Any]], JSONEnforcementReport]:
        """Validate LLM response and attempt fixes if needed"""
        
        enforcement_result = self.enforcer.enforce_json_output(llm_response, self.schema)
        
        if enforcement_result.success:
            return enforcement_result.final_response, enforcement_result
        else:
            logger.error(f"JSON validation failed: {enforcement_result.final_error}")
            return None, enforcement_result
    
    def create_fallback_response(self, error_report: JSONEnforcementReport, 
                                original_input: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a safe fallback response when JSON enforcement fails"""
        
        return {
            "scores": {
                "technical_accuracy": 3.0,
                "concept_clarity": 3.0,
                "keyword_coverage": 3.0,
                "communication": 3.0,
                "final_score": 3.0,
                "confidence_score": 1
            },
            "feedback": f"Evaluation failed due to response parsing error. {error_report.final_error}",
            "anti_cheat": {
                "is_copy_paste": False,
                "is_ai_generated": False,
                "is_too_robotic": False,
                "transcript_mismatch": False,
                "confidence_level": 0.0,
                "risk_factors": ["JSON parsing failure", "Fallback response used"]
            },
            "keyword_analysis": original_input.get("expected_keywords", {}) if original_input else {},
            "response_quality": "error",
            "areas_for_improvement": ["System encountered parsing error", "Please retry evaluation"],
            "processing_metadata": {
                "json_enforcement_failed": True,
                "total_attempts": error_report.total_attempts,
                "fixes_attempted": len(error_report.fixes_applied),
                "error_type": error_report.final_error,
                "fallback_used": True
            }
        }


# Utility functions for integration
def enforce_json_response(llm_response: str) -> Tuple[Optional[Dict[str, Any]], bool, str]:
    """
    Simple utility function to enforce JSON response
    
    Returns:
        (parsed_data, success, error_message)
    """
    validator = EnhancedJSONValidator()
    result, report = validator.validate_and_fix(llm_response)
    
    if result:
        return result, True, ""
    else:
        fallback = validator.create_fallback_response(report)
        return fallback, False, report.final_error or "Unknown JSON parsing error"


def test_json_enforcement():
    """Test function for JSON enforcement capabilities"""
    
    test_responses = [
        # Valid JSON
        '{"scores": {"technical_accuracy": 8.5}, "feedback": "Good answer"}',
        
        # JSON with markdown
        '```json\n{"scores": {"technical_accuracy": 7.0}, "feedback": "Average response"}\n```',
        
        # Malformed JSON with trailing commas
        '{"scores": {"technical_accuracy": 6.0,}, "feedback": "Needs improvement",}',
        
        # JSON with single quotes
        "{'scores': {'technical_accuracy': 5.0}, 'feedback': 'Poor answer'}",
        
        # Completely malformed
        'This is not JSON at all. The answer was good though.',
    ]
    
    enforcer = AdvancedJSONEnforcer()
    
    for i, response in enumerate(test_responses, 1):
        print(f"\n📝 Test Case {i}")
        print(f"Input: {response[:50]}...")
        
        result = enforcer.enforce_json_output(response)
        print(f"Success: {'✅' if result.success else '❌'}")
        print(f"Attempts: {result.total_attempts}")
        if result.fixes_applied:
            print(f"Fixes: {', '.join(result.fixes_applied)}")


if __name__ == "__main__":
    test_json_enforcement()