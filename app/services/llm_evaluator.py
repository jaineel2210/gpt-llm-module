# app/services/llm_evaluator.py

import os
import json
import time
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, AuthenticationError, Timeout

from app.config import (
    LLM_CONFIG, DEFAULT_WEIGHTS, ANTI_CHEAT_CONFIG, 
    RISK_LEVELS, ERROR_CONFIG, OPENAI_API_KEY, USE_MOCK_MODE
)
from app.schemas import (
    EvaluationInput, EvaluationOutput, EvaluationScores, 
    AntiCheatDetection, RiskEngineOutput, ScoringWeights, 
    ConfidenceAdjustmentResult, PlagiarismAnalysis, ContextAnalysis
)
from app.services.json_enforcer import EnhancedJSONValidator
from app.services.confidence_adjuster import AdvancedConfidenceAdjuster
from app.services.plagiarism_detector_simple import AdvancedPlagiarismDetector
from app.services.context_manager import MultiTurnContextManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = None
if OPENAI_API_KEY and not USE_MOCK_MODE:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")

# Load evaluation prompt template
PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "evaluation_prompt.txt"
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    EVALUATION_PROMPT_TEMPLATE = f.read()


class LLMEvaluator:
    """Advanced LLM Evaluation System with Anti-Cheating Detection"""
    
    def __init__(self, weights: Optional[ScoringWeights] = None):
        self.weights = weights or ScoringWeights()
        self.client = client
        self.json_validator = EnhancedJSONValidator()
        self.confidence_adjuster = AdvancedConfidenceAdjuster()
        self.plagiarism_detector = AdvancedPlagiarismDetector()
        self.context_manager = MultiTurnContextManager()
        self.current_template = EVALUATION_PROMPT_TEMPLATE
        
    def get_current_template(self) -> str:
        """Get the current prompt template"""
        return self.current_template
    
    def set_template(self, template: str):
        """Set a new prompt template for testing"""
        self.current_template = template
        
    def evaluate_answer(self, evaluation_input: EvaluationInput) -> EvaluationOutput:
        """
        Main evaluation function that coordinates all evaluation steps
        
        Args:
            evaluation_input: Comprehensive evaluation input data
            
        Returns:
            Structured evaluation output with scores, feedback, and risk analysis
        """
        try:
            # Step 1: Perform comprehensive evaluation with retries
            llm_response = self._evaluate_with_retries(evaluation_input)
            
            # Step 2: Parse and validate response
            parsed_response = self._parse_llm_response(llm_response)
            
            # Step 3: Apply business logic validation
            validated_response = self._apply_business_validation(
                parsed_response, evaluation_input
            )
            
            # Step 4: Add processing metadata
            validated_response = self._add_processing_metadata(
                validated_response, evaluation_input
            )
            
            return validated_response
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return self._generate_fallback_response(evaluation_input, str(e))
    
    def _evaluate_with_retries(self, evaluation_input: EvaluationInput) -> str:
        """Evaluate with retry logic and fallback handling"""
        
        if USE_MOCK_MODE or not self.client:
            return self._generate_mock_evaluation(evaluation_input)
        
        for attempt in range(ERROR_CONFIG["max_retries"]):
            try:
                # Get interview context if multi-turn is enabled
                interview_context = ""
                if (evaluation_input.enable_multi_turn_context and 
                    evaluation_input.interview_id):
                    context_info = self.context_manager.get_context_for_evaluation(
                        evaluation_input.interview_id
                    )
                    if context_info:
                        interview_context = f"\n\nINTERVIEW CONTEXT:\n{context_info}"
                
                # Format the comprehensive prompt using current template
                formatted_prompt = self.current_template.format(
                    question=evaluation_input.question,
                    candidate_answer=evaluation_input.candidate_answer,
                    expected_keywords=", ".join(evaluation_input.expected_keywords),
                    experience_level=evaluation_input.experience_level.value,
                    question_type=evaluation_input.question_type.value,
                    context=evaluation_input.context or "No additional context",
                    audio_transcript=evaluation_input.audio_transcript or "Not provided"
                ) + interview_context
                
                # Make API call with comprehensive configuration
                response = self.client.chat.completions.create(
                    model=LLM_CONFIG["model"],
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert AI interview evaluator. Respond with STRICT JSON only. NO additional text."
                        },
                        {"role": "user", "content": formatted_prompt}
                    ],
                    temperature=LLM_CONFIG["temperature"],
                    max_tokens=LLM_CONFIG["max_tokens"],
                    top_p=LLM_CONFIG["top_p"],
                    frequency_penalty=LLM_CONFIG["frequency_penalty"],
                    presence_penalty=LLM_CONFIG["presence_penalty"],
                    timeout=ERROR_CONFIG["timeout_seconds"]
                )
                
                return response.choices[0].message.content
                
            except (RateLimitError, APIError, AuthenticationError, Timeout) as e:
                logger.warning(f"API error on attempt {attempt + 1}: {str(e)}")
                if attempt == ERROR_CONFIG["max_retries"] - 1:
                    if ERROR_CONFIG["fallback_to_mock"]:
                        logger.info("Falling back to mock evaluation")
                        return self._generate_mock_evaluation(evaluation_input)
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
        
        # Should not reach here
        raise Exception("Max retries exceeded")
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM response JSON with enhanced enforcement"""
        
        # Use enhanced JSON validator
        parsed_response, enforcement_report = self.json_validator.validate_and_fix(response)
        
        if parsed_response:
            logger.info("JSON validation successful")
            if enforcement_report.fixes_applied:
                logger.info(f"Applied fixes: {', '.join(enforcement_report.fixes_applied)}")
            return parsed_response
        else:
            # Create fallback response
            logger.error("JSON validation failed, using fallback response")
            logger.debug(f"Enforcement report: {enforcement_report.final_error}")
            
            fallback = self.json_validator.create_fallback_response(enforcement_report)
            return fallback
    
    def _apply_business_validation(self, parsed_response: Dict[str, Any], 
                                 evaluation_input: EvaluationInput) -> EvaluationOutput:
        """Apply business logic validation and corrections"""
        
        # Validate and correct scores
        scores_data = parsed_response["scores"]
        
        # Ensure scores are within valid range
        for score_key in ["technical_accuracy", "concept_clarity", "keyword_coverage", "communication"]:
            scores_data[score_key] = max(0, min(10, float(scores_data[score_key])))
        
        # Recalculate final score using proper weights
        final_score = (
            scores_data["technical_accuracy"] * self.weights.technical_accuracy +
            scores_data["concept_clarity"] * self.weights.concept_clarity +
            scores_data["keyword_coverage"] * self.weights.keyword_coverage +
            scores_data["communication"] * self.weights.communication
        )
        scores_data["final_score"] = round(final_score, 2)
        
        # Apply confidence adjustment if confidence metrics are provided and enabled
        confidence_adjustment_result = None
        if (evaluation_input.enable_confidence_adjustment and 
            evaluation_input.confidence_metrics and 
            evaluation_input.confidence_metrics.whisper_confidence is not None):
            
            # Prepare original scores for adjustment
            original_scores = {
                "technical_accuracy": scores_data["technical_accuracy"],
                "concept_clarity": scores_data["concept_clarity"],
                "keyword_coverage": scores_data["keyword_coverage"],
                "communication": scores_data["communication"]
            }
            
            # Apply confidence adjustment
            confidence_result = self.confidence_adjuster.adjust_evaluation_scores(
                original_scores=original_scores,
                whisper_confidence=evaluation_input.confidence_metrics.whisper_confidence,
                audio_quality_score=evaluation_input.confidence_metrics.audio_quality_score,
                speech_pattern_consistency=evaluation_input.confidence_metrics.speech_pattern_consistency,
                background_noise_level=evaluation_input.confidence_metrics.background_noise_level,
                preserve_communication_score=True
            )
            
            # Update scores with adjusted values
            scores_data["technical_accuracy"] = confidence_result.adjusted_scores["technical_accuracy"]
            scores_data["concept_clarity"] = confidence_result.adjusted_scores["concept_clarity"]
            scores_data["keyword_coverage"] = confidence_result.adjusted_scores["keyword_coverage"]
            scores_data["communication"] = confidence_result.adjusted_scores["communication"]
            
            # Recalculate final score with adjusted values
            adjusted_final_score = (
                scores_data["technical_accuracy"] * self.weights.technical_accuracy +
                scores_data["concept_clarity"] * self.weights.concept_clarity +
                scores_data["keyword_coverage"] * self.weights.keyword_coverage +
                scores_data["communication"] * self.weights.communication
            )
            scores_data["final_score"] = round(adjusted_final_score, 2)
            
            # Create confidence adjustment result for output
            confidence_adjustment_result = ConfidenceAdjustmentResult(
                original_scores=confidence_result.original_scores,
                adjusted_scores=confidence_result.adjusted_scores,
                adjustment_factor=confidence_result.adjustment_factor,
                adjustment_reason=confidence_result.adjustment_reason,
                recommendation=confidence_result.recommendation,
                confidence_breakdown=confidence_result.confidence_metrics
            )
            
            logger.info(f"Applied confidence adjustment: factor={confidence_result.adjustment_factor:.3f}, "
                       f"reason='{confidence_result.adjustment_reason}'")
        
        # Validate confidence score
        scores_data["confidence_score"] = max(1, min(10, int(scores_data.get("confidence_score", 5))))
        
        # Create validated scores object
        validated_scores = EvaluationScores(**scores_data)
        
        # Enhance anti-cheat detection
        anti_cheat_data = parsed_response["anti_cheat"]
        enhanced_anti_cheat = self._enhance_anti_cheat_detection(
            anti_cheat_data, evaluation_input
        )
        
        # Perform detailed plagiarism analysis if enabled
        plagiarism_analysis_result = None
        if evaluation_input.enable_plagiarism_detection:
            try:
                plagiarism_result = self.plagiarism_detector.analyze_plagiarism(
                    candidate_answer=evaluation_input.candidate_answer,
                    question=evaluation_input.question,
                    expected_keywords=evaluation_input.expected_keywords,
                    experience_level=evaluation_input.experience_level.value
                )
                
                plagiarism_analysis_result = PlagiarismAnalysis(
                    risk_level=plagiarism_result.risk_level,
                    overall_similarity=plagiarism_result.overall_similarity,
                    confidence=plagiarism_result.confidence,
                    similarity_breakdown=plagiarism_result.similarity_metrics,
                    ideal_answer=plagiarism_result.ideal_answer,
                    flagged_sections=plagiarism_result.flagged_sections,
                    explanation=plagiarism_result.explanation,
                    recommendation=plagiarism_result.recommendation,
                    processing_time_ms=plagiarism_result.processing_time_ms
                )
                
                logger.info(f"Detailed plagiarism analysis completed: {plagiarism_result.risk_level} risk")
                
            except Exception as e:
                logger.warning(f"Detailed plagiarism analysis failed: {e}")
        
        # Handle multi-turn interview context if enabled
        context_analysis_result = None
        if (evaluation_input.enable_multi_turn_context and 
            evaluation_input.interview_id):
            try:
                # Update context with current Q&A
                keywords_covered = [kw for kw, found in parsed_response["keyword_analysis"].items() if found]
                
                context_state = self.context_manager.add_question_answer(
                    interview_id=evaluation_input.interview_id,
                    question=evaluation_input.question,
                    answer=evaluation_input.candidate_answer,
                    question_type=evaluation_input.question_type.value,
                    score=validated_scores.final_score,
                    keywords_covered=keywords_covered,
                    evaluation_metadata={
                        "anti_cheat_confidence": enhanced_anti_cheat.confidence_level,
                        "plagiarism_risk": enhanced_anti_cheat.plagiarism_risk,
                        "processing_timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                # Create context analysis result
                context_analysis_result = ContextAnalysis(
                    interview_id=evaluation_input.interview_id,
                    total_questions=context_state.total_questions,
                    consistency_level=context_state.consistency_analysis.overall_consistency if context_state.consistency_analysis else None,
                    consistency_score=context_state.consistency_analysis.consistency_score if context_state.consistency_analysis else None,
                    topic_progression=context_state.topic_progression,
                    context_summary=context_state.context_summary,
                    context_influence=self._determine_context_influence(context_state, validated_scores.final_score)
                )
                
                logger.info(f"Updated multi-turn context for interview {evaluation_input.interview_id}. "
                           f"Total questions: {context_state.total_questions}")
                
            except Exception as e:
                logger.warning(f"Multi-turn context processing failed: {e}")
        
        # Create comprehensive response
        evaluation_output = EvaluationOutput(
            scores=validated_scores,
            feedback=parsed_response["feedback"],
            anti_cheat=enhanced_anti_cheat,
            keyword_analysis=parsed_response["keyword_analysis"],
            response_quality=parsed_response["response_quality"],
            areas_for_improvement=parsed_response["areas_for_improvement"],
            confidence_adjustment=confidence_adjustment_result,
            plagiarism_analysis=plagiarism_analysis_result,
            context_analysis=context_analysis_result,
            processing_metadata=parsed_response["processing_metadata"]
        )
        
        return evaluation_output
    
    def _determine_context_influence(self, context_state, current_score: float) -> str:
        """
        Determine how the interview context influenced the current evaluation
        
        Args:
            context_state: Current context state
            current_score: Score for the current answer
            
        Returns:
            Description of context influence
        """
        if context_state.total_questions == 1:
            return "No previous context available for first question"
        
        # Get recent scores for comparison
        recent_scores = [qa_dict['score'] for qa_dict in context_state.question_history[-3:]]
        avg_recent_score = sum(recent_scores) / len(recent_scores)
        
        # Determine influence based on consistency and performance trends
        if context_state.consistency_analysis:
            consistency_level = context_state.consistency_analysis.overall_consistency
            
            if consistency_level in ["highly_consistent", "consistent"]:
                if abs(current_score - avg_recent_score) < 1.0:
                    return "Context supports consistent performance pattern"
                else:
                    return "Current performance deviates from established consistent pattern"
            
            elif consistency_level == "somewhat_consistent":
                return "Context shows mixed performance; current answer fits moderate consistency pattern"
            
            else:
                return "Context shows inconsistent performance; current answer adds to concerning pattern"
        
        else:
            if len(recent_scores) > 1:
                score_trend = recent_scores[-1] - recent_scores[0]
                if score_trend > 1.0:
                    return "Context shows improving trend; current performance continues upward trajectory"
                elif score_trend < -1.0:
                    return "Context shows declining trend; current performance continues downward pattern"
                else:
                    return "Context shows stable performance; current answer maintains consistent level"
            
            return "Limited context available; minimal influence on evaluation"
    
    def _enhance_anti_cheat_detection(self, base_detection: Dict[str, Any], 
                                    evaluation_input: EvaluationInput) -> AntiCheatDetection:
        """Enhance anti-cheat detection with additional heuristics"""
        
        answer = evaluation_input.candidate_answer.lower()
        risk_factors = list(base_detection.get("risk_factors", []))
        
        # Additional copy-paste detection
        copy_paste_indicators = ANTI_CHEAT_CONFIG["copy_paste_keywords"]
        for indicator in copy_paste_indicators:
            if indicator in answer:
                risk_factors.append(f"Copy-paste indicator: {indicator}")
                base_detection["is_copy_paste"] = True
        
        # Additional AI-generated content detection
        ai_indicators = ANTI_CHEAT_CONFIG["ai_generated_indicators"]
        for indicator in ai_indicators:
            if indicator in answer:
                risk_factors.append(f"AI-generated indicator: {indicator}")
                base_detection["is_ai_generated"] = True
        
        # Robotic response detection
        word_count = len(answer.split())
        if word_count < ANTI_CHEAT_CONFIG["min_answer_length"]:
            risk_factors.append("Answer too short")
        
        # Check for natural speech patterns
        contractions = ["don't", "can't", "won't", "i'm", "it's", "that's", "we're"]
        has_contractions = any(cont in answer for cont in contractions)
        if not has_contractions and word_count > 50:
            risk_factors.append("Lacks natural speech contractions")
            base_detection["is_too_robotic"] = True
        
        # Calculate enhanced confidence level
        risk_count = sum([
            base_detection.get("is_copy_paste", False),
            base_detection.get("is_ai_generated", False),
            base_detection.get("is_too_robotic", False),
            base_detection.get("transcript_mismatch", False)
        ])
        
        enhanced_confidence = min(1.0, base_detection.get("confidence_level", 0.5) + (risk_count * 0.2))
        
        # Initialize plagiarism detection fields
        plagiarism_risk = None
        similarity_score = None
        plagiarism_confidence = None
        
        # Perform plagiarism detection if enabled
        if evaluation_input.enable_plagiarism_detection:
            try:
                plagiarism_result = self.plagiarism_detector.analyze_plagiarism(
                    candidate_answer=evaluation_input.candidate_answer,
                    question=evaluation_input.question,
                    expected_keywords=evaluation_input.expected_keywords,
                    experience_level=evaluation_input.experience_level.value
                )
                
                plagiarism_risk = plagiarism_result.risk_level
                similarity_score = plagiarism_result.overall_similarity
                plagiarism_confidence = plagiarism_result.confidence
                
                # Add plagiarism-related risk factors
                if plagiarism_result.risk_level in ["high", "critical"]:
                    risk_factors.extend(plagiarism_result.risk_factors)
                    if plagiarism_result.overall_similarity > 0.8:
                        risk_factors.append(f"High similarity to ideal answer ({plagiarism_result.overall_similarity:.1%})")
                
                logger.info(f"Plagiarism detection completed: {plagiarism_result.risk_level} risk, "
                           f"{plagiarism_result.overall_similarity:.1%} similarity")
                
            except Exception as e:
                logger.error(f"Plagiarism detection failed: {e}")
                risk_factors.append("Plagiarism detection failed")
        
        return AntiCheatDetection(
            is_copy_paste=base_detection.get("is_copy_paste", False),
            is_ai_generated=base_detection.get("is_ai_generated", False),
            is_too_robotic=base_detection.get("is_too_robotic", False),
            transcript_mismatch=base_detection.get("transcript_mismatch", False),
            confidence_level=enhanced_confidence,
            risk_factors=risk_factors,
            plagiarism_risk=plagiarism_risk,
            similarity_score=similarity_score,
            plagiarism_confidence=plagiarism_confidence
        )
    
    def _add_processing_metadata(self, response: EvaluationOutput, 
                               evaluation_input: EvaluationInput) -> EvaluationOutput:
        """Add additional processing metadata"""
        
        # Calculate additional metrics
        word_count = len(evaluation_input.candidate_answer.split())
        response.processing_metadata.update({
            "evaluation_timestamp": datetime.utcnow().isoformat(),
            "word_count": word_count,
            "question_type": evaluation_input.question_type.value,
            "experience_level": evaluation_input.experience_level.value,
            "keyword_count": len(evaluation_input.expected_keywords),
            "processing_time_ms": int(time.time() * 1000) % 1000
        })
        
        return response
    
    def _generate_mock_evaluation(self, evaluation_input: EvaluationInput) -> str:
        """Generate sophisticated mock evaluation for testing"""
        
        answer_length = len(evaluation_input.candidate_answer.split())
        keyword_coverage = sum(1 for kw in evaluation_input.expected_keywords 
                             if kw.lower() in evaluation_input.candidate_answer.lower())
        
        # Calculate mock scores with some intelligence
        base_score = min(8.5, 5.0 + (answer_length / 30))
        technical_score = min(10.0, base_score + (keyword_coverage * 0.5))
        clarity_score = min(10.0, base_score * 0.9)
        keyword_score = (keyword_coverage / max(1, len(evaluation_input.expected_keywords))) * 10
        communication_score = min(10.0, base_score * 0.85)
        
        # Calculate weighted final score
        final_score = (
            technical_score * DEFAULT_WEIGHTS["technical_accuracy"] +
            clarity_score * DEFAULT_WEIGHTS["concept_clarity"] +
            keyword_score * DEFAULT_WEIGHTS["keyword_coverage"] +
            communication_score * DEFAULT_WEIGHTS["communication"]
        )
        
        mock_response = {
            "scores": {
                "technical_accuracy": round(technical_score, 1),
                "concept_clarity": round(clarity_score, 1),
                "keyword_coverage": round(keyword_score, 1),
                "communication": round(communication_score, 1),
                "final_score": round(final_score, 2),
                "confidence_score": 7
            },
            "feedback": f"Mock evaluation: The answer demonstrates {evaluation_input.experience_level.value}-level understanding. Shows good grasp of concepts with room for improvement in specific examples.",
            "anti_cheat": {
                "is_copy_paste": False,
                "is_ai_generated": False,
                "is_too_robotic": answer_length < 20,
                "transcript_mismatch": False,
                "confidence_level": 0.3,
                "risk_factors": ["This is mock evaluation"] if answer_length < 10 else []
            },
            "keyword_analysis": {kw: kw.lower() in evaluation_input.candidate_answer.lower() 
                               for kw in evaluation_input.expected_keywords},
            "response_quality": "good" if final_score > 7 else "fair",
            "areas_for_improvement": [
                "Add more specific examples",
                "Improve technical depth",
                "Better structure in explanation"
            ],
            "processing_metadata": {
                "answer_length": answer_length,
                "complexity_level": "medium",
                "domain_match": True
            }
        }
        
        return json.dumps(mock_response)
    
    def _generate_fallback_response(self, evaluation_input: EvaluationInput, 
                                  error_msg: str) -> EvaluationOutput:
        """Generate fallback response when all else fails"""
        
        return EvaluationOutput(
            scores=EvaluationScores(
                technical_accuracy=5.0,
                concept_clarity=5.0,
                keyword_coverage=5.0,
                communication=5.0,
                final_score=5.0,
                confidence_score=1
            ),
            feedback=f"Evaluation failed due to system error. Please retry. Error: {error_msg}",
            anti_cheat=AntiCheatDetection(
                is_copy_paste=False,
                is_ai_generated=False,
                is_too_robotic=False,
                transcript_mismatch=False,
                confidence_level=0.1,
                risk_factors=["System evaluation failure"]
            ),
            keyword_analysis={kw: False for kw in evaluation_input.expected_keywords},
            response_quality="unknown",
            areas_for_improvement=["System error occurred - please retry evaluation"],
            processing_metadata={
                "error": error_msg,
                "fallback_used": True,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    def generate_risk_engine_output(self, evaluation_output: EvaluationOutput) -> RiskEngineOutput:
        """
        Convert evaluation output to risk engine format for integration
        """
        
        # Calculate overall cheat probability
        cheat_indicators = [
            evaluation_output.anti_cheat.is_copy_paste,
            evaluation_output.anti_cheat.is_ai_generated,
            evaluation_output.anti_cheat.is_too_robotic,
            evaluation_output.anti_cheat.transcript_mismatch
        ]
        
        cheat_probability = sum(cheat_indicators) / len(cheat_indicators)
        cheat_probability = min(1.0, cheat_probability * evaluation_output.anti_cheat.confidence_level)
        
        # Determine risk flag
        risk_flag = cheat_probability > 0.5 or evaluation_output.scores.final_score < 3.0
        
        # Determine confidence level
        if evaluation_output.scores.confidence_score >= 8:
            confidence_level = "high"
        elif evaluation_output.scores.confidence_score >= 5:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # Prepare quality metrics
        quality_metrics = {
            "technical_accuracy": evaluation_output.scores.technical_accuracy,
            "concept_clarity": evaluation_output.scores.concept_clarity,
            "keyword_coverage": evaluation_output.scores.keyword_coverage,
            "communication": evaluation_output.scores.communication
        }
        
        return RiskEngineOutput(
            llm_score=evaluation_output.scores.final_score,
            risk_flag=risk_flag,
            confidence_level=confidence_level,
            cheat_probability=cheat_probability,
            quality_metrics=quality_metrics,
            evaluation_timestamp=datetime.utcnow().isoformat(),
            metadata={
                "response_quality": evaluation_output.response_quality,
                "risk_factors": evaluation_output.anti_cheat.risk_factors,
                "processing_metadata": evaluation_output.processing_metadata
            }
        )


# Convenient functions for backward compatibility and easy usage
def evaluate_interview_answer(question: str, answer: str, experience_level: str) -> str:
    """Legacy function for backward compatibility"""
    
    try:
        evaluator = LLMEvaluator()
        
        # Convert to new input format
        evaluation_input = EvaluationInput(
            question=question,
            candidate_answer=answer,
            expected_keywords=[],  # Empty for legacy calls
            experience_level=experience_level.lower(),
            question_type="technical"  # Default
        )
        
        # Perform evaluation
        result = evaluator.evaluate_answer(evaluation_input)
        
        # Convert to legacy format
        legacy_response = {
            "relevance_score": result.scores.technical_accuracy,
            "clarity_score": result.scores.concept_clarity,
            "technical_accuracy": result.scores.technical_accuracy,
            "communication_score": result.scores.communication,
            "overall_score": result.scores.final_score,
            "feedback": result.feedback
        }
        
        return json.dumps(legacy_response)
        
    except Exception as e:
        logger.error(f"Legacy evaluation failed: {str(e)}")
        # Return basic mock response
        return json.dumps({
            "relevance_score": 5.0,
            "clarity_score": 5.0,
            "technical_accuracy": 5.0,
            "communication_score": 5.0,
            "overall_score": 5.0,
            "feedback": "Evaluation failed - please retry"
        })
