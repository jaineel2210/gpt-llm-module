"""
Advanced Confidence Adjustment Layer for LLM Interview Evaluation System

This module implements intelligent confidence-based evaluation weight adjustments
that account for speech recognition accuracy and audio quality issues.
Integrates with Whisper confidence scores to ensure fair evaluation.
"""

from typing import Dict, Tuple, Optional, List, Any
import logging
from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceMetrics:
    """Data class for tracking confidence-related metrics"""
    whisper_confidence: float
    audio_quality_score: float
    speech_pattern_consistency: float
    background_noise_level: float
    overall_confidence: float


class ConfidenceAdjustmentConfig(BaseModel):
    """Configuration for confidence adjustment parameters"""
    
    # Base thresholds
    critical_threshold: float = Field(default=0.40, ge=0.0, le=1.0, description="Below this = major adjustment")
    low_threshold: float = Field(default=0.60, ge=0.0, le=1.0, description="Below this = moderate adjustment")
    good_threshold: float = Field(default=0.80, ge=0.0, le=1.0, description="Above this = minimal adjustment")
    
    # Adjustment factors (multipliers for final scores)
    critical_multiplier: float = Field(default=0.50, ge=0.1, le=1.0, description="Severe confidence issues")
    low_multiplier: float = Field(default=0.75, ge=0.1, le=1.0, description="Moderate confidence issues")
    moderate_multiplier: float = Field(default=0.90, ge=0.1, le=1.0, description="Minor confidence issues")
    
    # Component weights for overall confidence calculation
    whisper_weight: float = Field(default=0.50, ge=0.0, le=1.0, description="Whisper confidence importance")
    audio_quality_weight: float = Field(default=0.25, ge=0.0, le=1.0, description="Audio quality importance")
    speech_pattern_weight: float = Field(default=0.15, ge=0.0, le=1.0, description="Speech pattern consistency")
    background_noise_weight: float = Field(default=0.10, ge=0.0, le=1.0, description="Background noise impact")
    
    # Smoothing parameters
    enable_smoothing: bool = Field(default=True, description="Enable confidence smoothing")
    smoothing_window: int = Field(default=5, ge=1, le=20, description="Window size for confidence smoothing")
    gradient_adjustment: bool = Field(default=True, description="Use gradient-based adjustments")
    
    @validator('whisper_weight', 'audio_quality_weight', 'speech_pattern_weight', 'background_noise_weight')
    def check_weights_sum_to_one(cls, v, values):
        """Ensure all weights sum approximately to 1.0"""
        if 'background_noise_weight' in values:
            total = (values.get('whisper_weight', 0) + 
                    values.get('audio_quality_weight', 0) + 
                    values.get('speech_pattern_weight', 0) + v)
            if abs(total - 1.0) > 0.01:
                logger.warning(f"Confidence weights sum to {total:.3f}, not 1.0. Auto-adjusting.")
        return v


class ConfidenceAdjustmentResult(BaseModel):
    """Result of confidence adjustment process"""
    
    original_scores: Dict[str, float]
    adjusted_scores: Dict[str, float]
    confidence_metrics: Dict[str, float]
    adjustment_factor: float
    adjustment_reason: str
    recommendation: str
    metadata: Dict[str, Any]


class AdvancedConfidenceAdjuster:
    """
    Advanced confidence adjustment system that modifies evaluation scores
    based on speech recognition confidence and audio quality metrics.
    """
    
    def __init__(self, config: Optional[ConfidenceAdjustmentConfig] = None):
        """
        Initialize the confidence adjuster
        
        Args:
            config: Configuration for adjustment parameters
        """
        self.config = config or ConfidenceAdjustmentConfig()
        self.confidence_history: List[float] = []
        self.adjustment_history: List[Dict] = []
        
        logger.info("Initialized Advanced Confidence Adjuster")
        logger.info(f"Thresholds: Critical={self.config.critical_threshold:.2f}, "
                   f"Low={self.config.low_threshold:.2f}, Good={self.config.good_threshold:.2f}")
    
    def calculate_overall_confidence(self, 
                                   whisper_confidence: float,
                                   audio_quality_score: Optional[float] = None,
                                   speech_pattern_consistency: Optional[float] = None,
                                   background_noise_level: Optional[float] = None) -> ConfidenceMetrics:
        """
        Calculate overall confidence from multiple metrics
        
        Args:
            whisper_confidence: Whisper ASR confidence score (0.0-1.0)
            audio_quality_score: Audio quality assessment (0.0-1.0)
            speech_pattern_consistency: Speech pattern consistency (0.0-1.0)
            background_noise_level: Background noise level (0.0=clean, 1.0=very noisy)
        
        Returns:
            ConfidenceMetrics with calculated overall confidence
        """
        # Use defaults for missing metrics
        audio_quality = audio_quality_score if audio_quality_score is not None else 0.8
        speech_consistency = speech_pattern_consistency if speech_pattern_consistency is not None else 0.8
        noise_level = background_noise_level if background_noise_level is not None else 0.2
        
        # Convert noise level to positive score (less noise = higher score)
        noise_score = 1.0 - noise_level
        
        # Calculate weighted overall confidence
        overall_confidence = (
            whisper_confidence * self.config.whisper_weight +
            audio_quality * self.config.audio_quality_weight +
            speech_consistency * self.config.speech_pattern_weight +
            noise_score * self.config.background_noise_weight
        )
        
        # Apply smoothing if enabled
        if self.config.enable_smoothing and len(self.confidence_history) > 0:
            overall_confidence = self._apply_smoothing(overall_confidence)
        
        # Store in history
        self.confidence_history.append(overall_confidence)
        if len(self.confidence_history) > self.config.smoothing_window:
            self.confidence_history.pop(0)
        
        return ConfidenceMetrics(
            whisper_confidence=whisper_confidence,
            audio_quality_score=audio_quality,
            speech_pattern_consistency=speech_consistency,
            background_noise_level=noise_level,
            overall_confidence=overall_confidence
        )
    
    def _apply_smoothing(self, current_confidence: float) -> float:
        """Apply temporal smoothing to confidence scores"""
        if len(self.confidence_history) < 2:
            return current_confidence
        
        # Simple exponential moving average
        alpha = 0.3  # Smoothing factor
        previous_avg = np.mean(self.confidence_history[-3:])
        smoothed = alpha * current_confidence + (1 - alpha) * previous_avg
        
        return smoothed
    
    def determine_adjustment_factor(self, confidence_metrics: ConfidenceMetrics) -> Tuple[float, str, str]:
        """
        Determine the adjustment factor based on confidence metrics
        
        Args:
            confidence_metrics: Calculated confidence metrics
        
        Returns:
            Tuple of (adjustment_factor, reason, recommendation)
        """
        confidence = confidence_metrics.overall_confidence
        whisper_conf = confidence_metrics.whisper_confidence
        
        # Determine adjustment category
        if confidence < self.config.critical_threshold:
            factor = self.config.critical_multiplier
            reason = f"Critical confidence issues (overall: {confidence:.2f}, whisper: {whisper_conf:.2f})"
            recommendation = "Consider manual review or re-recording"
            
        elif confidence < self.config.low_threshold:
            factor = self.config.low_multiplier
            reason = f"Low confidence detected (overall: {confidence:.2f}, whisper: {whisper_conf:.2f})"
            recommendation = "Moderate confidence reduction applied"
            
        elif confidence < self.config.good_threshold:
            factor = self.config.moderate_multiplier
            reason = f"Moderate confidence (overall: {confidence:.2f}, whisper: {whisper_conf:.2f})"
            recommendation = "Minor confidence adjustment applied"
            
        else:
            factor = 1.0  # No adjustment needed
            reason = f"Good confidence (overall: {confidence:.2f}, whisper: {whisper_conf:.2f})"
            recommendation = "No adjustment needed"
        
        # Apply gradient adjustment for smoother transitions
        if self.config.gradient_adjustment and factor < 1.0:
            factor = self._apply_gradient_adjustment(confidence, factor)
        
        return factor, reason, recommendation
    
    def _apply_gradient_adjustment(self, confidence: float, base_factor: float) -> float:
        """
        Apply gradient-based adjustment for smoother transitions
        
        Args:
            confidence: Overall confidence score
            base_factor: Base adjustment factor
        
        Returns:
            Adjusted factor with smooth transitions
        """
        if confidence >= self.config.good_threshold:
            return 1.0
        
        # Linear interpolation for smooth transitions
        if confidence >= self.config.low_threshold:
            # Between low and good threshold
            progress = (confidence - self.config.low_threshold) / (self.config.good_threshold - self.config.low_threshold)
            return self.config.moderate_multiplier + progress * (1.0 - self.config.moderate_multiplier)
        
        elif confidence >= self.config.critical_threshold:
            # Between critical and low threshold
            progress = (confidence - self.config.critical_threshold) / (self.config.low_threshold - self.config.critical_threshold)
            return self.config.critical_multiplier + progress * (self.config.low_multiplier - self.config.critical_multiplier)
        
        else:
            # Below critical threshold
            return self.config.critical_multiplier
    
    def adjust_evaluation_scores(self,
                                original_scores: Dict[str, float],
                                whisper_confidence: float,
                                audio_quality_score: Optional[float] = None,
                                speech_pattern_consistency: Optional[float] = None,
                                background_noise_level: Optional[float] = None,
                                preserve_communication_score: bool = True) -> ConfidenceAdjustmentResult:
        """
        Adjust evaluation scores based on confidence metrics
        
        Args:
            original_scores: Original evaluation scores
            whisper_confidence: Whisper ASR confidence
            audio_quality_score: Audio quality assessment
            speech_pattern_consistency: Speech pattern consistency
            background_noise_level: Background noise level
            preserve_communication_score: Whether to preserve communication scores
        
        Returns:
            ConfidenceAdjustmentResult with adjusted scores
        """
        # Calculate confidence metrics
        confidence_metrics = self.calculate_overall_confidence(
            whisper_confidence=whisper_confidence,
            audio_quality_score=audio_quality_score,
            speech_pattern_consistency=speech_pattern_consistency,
            background_noise_level=background_noise_level
        )
        
        # Determine adjustment factor
        adjustment_factor, reason, recommendation = self.determine_adjustment_factor(confidence_metrics)
        
        # Apply adjustments
        adjusted_scores = {}
        
        for score_type, original_score in original_scores.items():
            if score_type == "communication" and preserve_communication_score:
                # Don't adjust communication scores for audio issues
                adjusted_scores[score_type] = original_score
            elif score_type in ["technical_accuracy", "concept_clarity", "keyword_coverage"]:
                # Apply full adjustment to content-based scores
                adjusted_scores[score_type] = original_score * adjustment_factor
            else:
                # Apply full adjustment to other scores
                adjusted_scores[score_type] = original_score * adjustment_factor
        
        # Calculate overall impact
        original_avg = np.mean(list(original_scores.values()))
        adjusted_avg = np.mean(list(adjusted_scores.values()))
        impact_percentage = ((adjusted_avg - original_avg) / original_avg) * 100
        
        # Store adjustment history
        adjustment_record = {
            "timestamp": None,  # Will be set by caller
            "confidence": confidence_metrics.overall_confidence,
            "adjustment_factor": adjustment_factor,
            "impact_percentage": impact_percentage,
            "reason": reason
        }
        self.adjustment_history.append(adjustment_record)
        
        # Keep history manageable
        if len(self.adjustment_history) > 100:
            self.adjustment_history.pop(0)
        
        return ConfidenceAdjustmentResult(
            original_scores=original_scores,
            adjusted_scores=adjusted_scores,
            confidence_metrics={
                "whisper_confidence": confidence_metrics.whisper_confidence,
                "audio_quality_score": confidence_metrics.audio_quality_score,
                "speech_pattern_consistency": confidence_metrics.speech_pattern_consistency,
                "background_noise_level": confidence_metrics.background_noise_level,
                "overall_confidence": confidence_metrics.overall_confidence
            },
            adjustment_factor=adjustment_factor,
            adjustment_reason=reason,
            recommendation=recommendation,
            metadata={
                "config_used": self.config.dict(),
                "impact_percentage": impact_percentage,
                "scores_preserved": ["communication"] if preserve_communication_score else [],
                "smoothing_applied": self.config.enable_smoothing,
                "gradient_adjustment_applied": self.config.gradient_adjustment
            }
        )
    
    def get_adjustment_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about recent adjustments
        
        Returns:
            Dictionary with adjustment statistics
        """
        if not self.adjustment_history:
            return {"message": "No adjustments recorded yet"}
        
        recent_adjustments = self.adjustment_history[-20:]  # Last 20 adjustments
        
        adjustment_factors = [adj["adjustment_factor"] for adj in recent_adjustments]
        confidences = [adj["confidence"] for adj in recent_adjustments]
        impacts = [adj["impact_percentage"] for adj in recent_adjustments]
        
        return {
            "total_adjustments": len(self.adjustment_history),
            "recent_average_confidence": np.mean(confidences),
            "average_adjustment_factor": np.mean(adjustment_factors),
            "average_impact_percentage": np.mean(impacts),
            "adjustments_by_severity": {
                "critical": sum(1 for f in adjustment_factors if f <= self.config.critical_multiplier + 0.05),
                "low": sum(1 for f in adjustment_factors if self.config.critical_multiplier < f <= self.config.low_multiplier + 0.05),
                "moderate": sum(1 for f in adjustment_factors if self.config.low_multiplier < f <= self.config.moderate_multiplier + 0.05),
                "none": sum(1 for f in adjustment_factors if f > self.config.moderate_multiplier + 0.05)
            },
            "configuration": {
                "critical_threshold": self.config.critical_threshold,
                "low_threshold": self.config.low_threshold,
                "good_threshold": self.config.good_threshold,
                "smoothing_enabled": self.config.enable_smoothing
            }
        }
    
    def update_configuration(self, new_config: ConfidenceAdjustmentConfig) -> None:
        """
        Update configuration parameters
        
        Args:
            new_config: New configuration to apply
        """
        old_config = self.config
        self.config = new_config
        
        logger.info("Updated confidence adjustment configuration")
        logger.info(f"Critical threshold: {old_config.critical_threshold:.2f} → {new_config.critical_threshold:.2f}")
        logger.info(f"Low threshold: {old_config.low_threshold:.2f} → {new_config.low_threshold:.2f}")
        logger.info(f"Good threshold: {old_config.good_threshold:.2f} → {new_config.good_threshold:.2f}")


# Example usage and testing functions
def example_usage():
    """Demonstrate confidence adjustment functionality"""
    
    # Initialize adjuster with custom config
    config = ConfidenceAdjustmentConfig(
        critical_threshold=0.4,
        low_threshold=0.6,
        good_threshold=0.8,
        critical_multiplier=0.5,
        low_multiplier=0.75,
        moderate_multiplier=0.9
    )
    
    adjuster = AdvancedConfidenceAdjuster(config)
    
    # Example evaluation scores
    original_scores = {
        "technical_accuracy": 8.5,
        "concept_clarity": 7.8,
        "keyword_coverage": 8.0,
        "communication": 7.2
    }
    
    # Case 1: Good confidence - no adjustment needed
    result1 = adjuster.adjust_evaluation_scores(
        original_scores=original_scores,
        whisper_confidence=0.87,
        audio_quality_score=0.9,
        speech_pattern_consistency=0.85,
        background_noise_level=0.1
    )
    
    print("Case 1 - Good Confidence:")
    print(f"  Adjustment Factor: {result1.adjustment_factor:.3f}")
    print(f"  Recommendation: {result1.recommendation}")
    print(f"  Original Score: {np.mean(list(result1.original_scores.values())):.2f}")
    print(f"  Adjusted Score: {np.mean(list(result1.adjusted_scores.values())):.2f}")
    
    # Case 2: Low confidence - moderate adjustment
    result2 = adjuster.adjust_evaluation_scores(
        original_scores=original_scores,
        whisper_confidence=0.45,  # Below 60% threshold
        audio_quality_score=0.6,
        speech_pattern_consistency=0.7,
        background_noise_level=0.4
    )
    
    print("\nCase 2 - Low Confidence:")
    print(f"  Adjustment Factor: {result2.adjustment_factor:.3f}")
    print(f"  Recommendation: {result2.recommendation}")
    print(f"  Original Score: {np.mean(list(result2.original_scores.values())):.2f}")
    print(f"  Adjusted Score: {np.mean(list(result2.adjusted_scores.values())):.2f}")
    
    # Case 3: Critical confidence - major adjustment
    result3 = adjuster.adjust_evaluation_scores(
        original_scores=original_scores,
        whisper_confidence=0.25,  # Very low
        audio_quality_score=0.4,
        speech_pattern_consistency=0.3,
        background_noise_level=0.8
    )
    
    print("\nCase 3 - Critical Confidence:")
    print(f"  Adjustment Factor: {result3.adjustment_factor:.3f}")
    print(f"  Recommendation: {result3.recommendation}")
    print(f"  Original Score: {np.mean(list(result3.original_scores.values())):.2f}")
    print(f"  Adjusted Score: {np.mean(list(result3.adjusted_scores.values())):.2f}")
    
    # Get adjustment statistics
    stats = adjuster.get_adjustment_statistics()
    print("\nAdjustment Statistics:")
    print(f"  Total Adjustments: {stats['total_adjustments']}")
    print(f"  Average Confidence: {stats['recent_average_confidence']:.3f}")
    print(f"  Average Adjustment Factor: {stats['average_adjustment_factor']:.3f}")


if __name__ == "__main__":
    example_usage()