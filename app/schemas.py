from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class ExperienceLevel(str, Enum):
    FRESHER = "fresher"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class QuestionType(str, Enum):
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    SYSTEM_DESIGN = "system_design"
    CODING = "coding"
    CONCEPTUAL = "conceptual"

class ConfidenceMetrics(BaseModel):
    """Confidence metrics from speech recognition and audio processing"""
    whisper_confidence: Optional[float] = Field(None, ge=0, le=1, description="Whisper ASR confidence score (0-1)")
    audio_quality_score: Optional[float] = Field(None, ge=0, le=1, description="Audio quality assessment (0-1)")
    speech_pattern_consistency: Optional[float] = Field(None, ge=0, le=1, description="Speech pattern consistency (0-1)")
    background_noise_level: Optional[float] = Field(None, ge=0, le=1, description="Background noise level (0=clean, 1=noisy)")
    overall_confidence: Optional[float] = Field(None, ge=0, le=1, description="Calculated overall confidence")

class EvaluationInput(BaseModel):
    """Comprehensive input format for LLM evaluation"""
    question: str = Field(..., description="The interview question asked")
    candidate_answer: str = Field(..., description="The candidate's response")
    expected_keywords: List[str] = Field(..., description="Key terms/concepts expected in answer")
    experience_level: ExperienceLevel = Field(..., description="Candidate's experience level")
    question_type: QuestionType = Field(..., description="Type of question being asked")
    context: Optional[str] = Field(None, description="Additional context about the role/domain")
    max_score: int = Field(10, description="Maximum possible score (default 10)")
    time_taken: Optional[int] = Field(None, description="Time taken to answer in seconds")
    audio_transcript: Optional[str] = Field(None, description="Original speech transcript if available")
    confidence_metrics: Optional[ConfidenceMetrics] = Field(None, description="Speech recognition confidence data")
    enable_confidence_adjustment: bool = Field(True, description="Whether to apply confidence-based adjustments")
    enable_plagiarism_detection: bool = Field(True, description="Whether to perform plagiarism similarity check")
    enable_multi_turn_context: bool = Field(True, description="Whether to use multi-turn interview context")
    interview_id: Optional[str] = Field(None, description="Interview session identifier for context tracking")
    candidate_id: Optional[str] = Field(None, description="Candidate identifier for context tracking")

class ScoringWeights(BaseModel):
    """Configurable scoring weights for different parameters"""
    technical_accuracy: float = Field(0.40, description="Weight for technical accuracy (40%)")
    concept_clarity: float = Field(0.25, description="Weight for concept clarity (25%)")
    keyword_coverage: float = Field(0.20, description="Weight for keyword coverage (20%)")
    communication: float = Field(0.15, description="Weight for communication (15%)")

class AntiCheatDetection(BaseModel):
    """Anti-cheating analysis results"""
    is_copy_paste: bool = Field(..., description="Detected copy-paste from textbooks")
    is_ai_generated: bool = Field(..., description="Detected AI-generated response")
    is_too_robotic: bool = Field(..., description="Response lacks natural speech patterns")
    transcript_mismatch: bool = Field(..., description="Mismatch between speech and text")
    confidence_level: float = Field(..., description="Overall confidence in detection (0-1)")
    risk_factors: List[str] = Field(..., description="List of identified risk factors")
    
    # Plagiarism detection results
    plagiarism_risk: Optional[str] = Field(None, description="Plagiarism risk level: none/low/medium/high/critical")
    similarity_score: Optional[float] = Field(None, ge=0, le=1, description="Overall similarity with ideal answer")
    plagiarism_confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence in plagiarism detection")

class EvaluationScores(BaseModel):
    """Detailed scoring breakdown"""
    technical_accuracy: float = Field(..., ge=0, le=10, description="Technical correctness score (0-10)")
    concept_clarity: float = Field(..., ge=0, le=10, description="Clarity of explanation (0-10)")
    keyword_coverage: float = Field(..., ge=0, le=10, description="Coverage of expected keywords (0-10)")
    communication: float = Field(..., ge=0, le=10, description="Communication effectiveness (0-10)")
    final_score: float = Field(..., ge=0, le=10, description="Weighted final score (0-10)")
    confidence_score: int = Field(..., ge=1, le=10, description="Evaluator confidence level (1-10)")

class ConfidenceAdjustmentResult(BaseModel):
    """Results of confidence-based score adjustment"""
    original_scores: Dict[str, float] = Field(..., description="Original evaluation scores")
    adjusted_scores: Dict[str, float] = Field(..., description="Confidence-adjusted scores") 
    adjustment_factor: float = Field(..., ge=0, le=1, description="Overall adjustment factor applied")
    adjustment_reason: str = Field(..., description="Reason for the adjustment")
    recommendation: str = Field(..., description="Recommendation based on confidence analysis")
    confidence_breakdown: Dict[str, float] = Field(..., description="Breakdown of confidence metrics")

class PlagiarismAnalysis(BaseModel):
    """Detailed plagiarism analysis results"""
    risk_level: str = Field(..., description="Risk level: none/low/medium/high/critical")
    overall_similarity: float = Field(..., ge=0, le=1, description="Overall similarity score")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    similarity_breakdown: Dict[str, float] = Field(..., description="Individual similarity metrics")
    ideal_answer: str = Field(..., description="Generated ideal answer for comparison")
    flagged_sections: List[Dict[str, Any]] = Field(..., description="Sections with high similarity")
    explanation: str = Field(..., description="Detailed explanation of findings")
    recommendation: str = Field(..., description="Recommended action")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class ContextAnalysis(BaseModel):
    """Multi-turn interview context analysis results"""
    interview_id: str = Field(..., description="Interview session identifier")
    total_questions: int = Field(..., description="Total questions in this interview")
    consistency_level: Optional[str] = Field(None, description="Overall consistency level")
    consistency_score: Optional[float] = Field(None, ge=0, le=1, description="Consistency score (0-1)")
    topic_progression: List[str] = Field(..., description="Topics covered in interview")
    context_summary: str = Field(..., description="Summary of interview context")
    context_influence: str = Field(..., description="How context influenced this evaluation")

class EvaluationOutput(BaseModel):
    """Comprehensive LLM evaluation output"""
    scores: EvaluationScores = Field(..., description="Detailed scoring breakdown")
    feedback: str = Field(..., description="Detailed feedback and suggestions")
    anti_cheat: AntiCheatDetection = Field(..., description="Anti-cheating analysis")
    keyword_analysis: Dict[str, bool] = Field(..., description="Which keywords were covered")
    response_quality: str = Field(..., description="Overall quality assessment")
    areas_for_improvement: List[str] = Field(..., description="Specific improvement suggestions")
    confidence_adjustment: Optional[ConfidenceAdjustmentResult] = Field(None, description="Confidence adjustment details")
    plagiarism_analysis: Optional[PlagiarismAnalysis] = Field(None, description="Detailed plagiarism analysis")
    context_analysis: Optional[ContextAnalysis] = Field(None, description="Multi-turn context analysis")
    processing_metadata: Dict[str, Any] = Field(..., description="Processing information")

class RiskEngineOutput(BaseModel):
    """Output format for integration with risk engine"""
    llm_score: float = Field(..., ge=0, le=10, description="Final LLM evaluation score")
    risk_flag: bool = Field(..., description="Whether answer poses cheating risk")
    confidence_level: str = Field(..., description="Confidence level: low/medium/high")
    cheat_probability: float = Field(..., ge=0, le=1, description="Probability of cheating (0-1)")
    quality_metrics: Dict[str, float] = Field(..., description="Individual quality scores")
    evaluation_timestamp: str = Field(..., description="When evaluation was performed")
    metadata: Dict[str, Any] = Field(..., description="Additional processing metadata")
