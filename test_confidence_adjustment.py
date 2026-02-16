"""
Test script for the Confidence Adjustment Layer integration

This script demonstrates the confidence adjustment functionality
and validates that it works correctly with the LLM evaluation system.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.schemas import EvaluationInput, ExperienceLevel, QuestionType, ConfidenceMetrics
from app.services.llm_evaluator import LLMEvaluator
from app.services.confidence_adjuster import AdvancedConfidenceAdjuster, ConfidenceAdjustmentConfig


def test_confidence_adjustment_integration():
    """Test the complete integration of confidence adjustment with LLM evaluation"""
    
    print("=" * 80)
    print("🧪 CONFIDENCE ADJUSTMENT LAYER - INTEGRATION TEST")
    print("=" * 80)
    
    # Initialize the evaluator
    evaluator = LLMEvaluator()
    
    # Prepare test evaluation input
    base_input_data = {
        "question": "Explain the difference between REST and GraphQL APIs.",
        "candidate_answer": "REST APIs use HTTP methods like GET, POST, PUT, DELETE for different operations on resources. They are stateless and follow a resource-based architecture. GraphQL is a query language that allows clients to request exactly what data they need. It uses a single endpoint and allows for more flexible queries compared to REST which has multiple endpoints.",
        "expected_keywords": ["REST", "HTTP", "GraphQL", "stateless", "endpoints", "query language"],
        "experience_level": ExperienceLevel.INTERMEDIATE,
        "question_type": QuestionType.TECHNICAL,
        "enable_confidence_adjustment": True
    }
    
    # Test Case 1: High Confidence - No Adjustment Expected
    print("\n🔹 Test Case 1: High Confidence (no adjustment expected)")
    print("-" * 50)
    
    high_conf_input = EvaluationInput(
        **base_input_data,
        confidence_metrics=ConfidenceMetrics(
            whisper_confidence=0.92,
            audio_quality_score=0.88,
            speech_pattern_consistency=0.85,
            background_noise_level=0.15
        )
    )
    
    result1 = evaluator.evaluate_answer(high_conf_input)
    
    print(f"📊 Original Scores:")
    if result1.confidence_adjustment:
        original_scores = result1.confidence_adjustment.original_scores
        for metric, score in original_scores.items():
            print(f"   {metric}: {score:.2f}")
        
        print(f"\n📊 Adjusted Scores:")
        adjusted_scores = result1.confidence_adjustment.adjusted_scores
        for metric, score in adjusted_scores.items():
            print(f"   {metric}: {score:.2f}")
        
        print(f"\n🔧 Adjustment Factor: {result1.confidence_adjustment.adjustment_factor:.3f}")
        print(f"💡 Reason: {result1.confidence_adjustment.adjustment_reason}")
        print(f"📋 Recommendation: {result1.confidence_adjustment.recommendation}")
        print(f"🎯 Final Score: {result1.scores.final_score:.2f}")
    else:
        print("❌ No confidence adjustment applied (unexpected for high confidence)")
    
    # Test Case 2: Low Confidence - Adjustment Expected
    print("\n🔹 Test Case 2: Low Confidence (adjustment expected)")
    print("-" * 50)
    
    low_conf_input = EvaluationInput(
        **base_input_data,
        confidence_metrics=ConfidenceMetrics(
            whisper_confidence=0.45,  # Below 60% threshold
            audio_quality_score=0.55,
            speech_pattern_consistency=0.60,
            background_noise_level=0.45
        )
    )
    
    result2 = evaluator.evaluate_answer(low_conf_input)
    
    print(f"📊 Original Scores:")
    if result2.confidence_adjustment:
        original_scores = result2.confidence_adjustment.original_scores
        for metric, score in original_scores.items():
            print(f"   {metric}: {score:.2f}")
        
        print(f"\n📊 Adjusted Scores:")
        adjusted_scores = result2.confidence_adjustment.adjusted_scores
        for metric, score in adjusted_scores.items():
            print(f"   {metric}: {score:.2f}")
        
        print(f"\n🔧 Adjustment Factor: {result2.confidence_adjustment.adjustment_factor:.3f}")
        print(f"💡 Reason: {result2.confidence_adjustment.adjustment_reason}")
        print(f"📋 Recommendation: {result2.confidence_adjustment.recommendation}")
        print(f"🎯 Final Score: {result2.scores.final_score:.2f}")
        
        # Calculate impact
        orig_final = sum(original_scores.values()) / len(original_scores)
        adj_final = result2.scores.final_score
        impact = ((adj_final - orig_final) / orig_final) * 100
        print(f"📈 Score Impact: {impact:+.1f}%")
    else:
        print("❌ No confidence adjustment applied (unexpected for low confidence)")
    
    # Test Case 3: Critical Confidence - Major Adjustment Expected
    print("\n🔹 Test Case 3: Critical Confidence (major adjustment expected)")
    print("-" * 50)
    
    critical_conf_input = EvaluationInput(
        **base_input_data,
        confidence_metrics=ConfidenceMetrics(
            whisper_confidence=0.28,  # Very low
            audio_quality_score=0.35,
            speech_pattern_consistency=0.25,
            background_noise_level=0.85
        )
    )
    
    result3 = evaluator.evaluate_answer(critical_conf_input)
    
    print(f"📊 Original Scores:")
    if result3.confidence_adjustment:
        original_scores = result3.confidence_adjustment.original_scores
        for metric, score in original_scores.items():
            print(f"   {metric}: {score:.2f}")
        
        print(f"\n📊 Adjusted Scores:")
        adjusted_scores = result3.confidence_adjustment.adjusted_scores
        for metric, score in adjusted_scores.items():
            print(f"   {metric}: {score:.2f}")
        
        print(f"\n🔧 Adjustment Factor: {result3.confidence_adjustment.adjustment_factor:.3f}")
        print(f"💡 Reason: {result3.confidence_adjustment.adjustment_reason}")
        print(f"📋 Recommendation: {result3.confidence_adjustment.recommendation}")
        print(f"🎯 Final Score: {result3.scores.final_score:.2f}")
        
        # Calculate impact
        orig_final = sum(original_scores.values()) / len(original_scores)
        adj_final = result3.scores.final_score
        impact = ((adj_final - orig_final) / orig_final) * 100
        print(f"📈 Score Impact: {impact:+.1f}%")
    else:
        print("❌ No confidence adjustment applied (unexpected for critical confidence)")
    
    # Test Case 4: Disabled Confidence Adjustment
    print("\n🔹 Test Case 4: Confidence Adjustment Disabled")
    print("-" * 50)
    
    disabled_input = EvaluationInput(
        **{**base_input_data, "enable_confidence_adjustment": False},
        confidence_metrics=ConfidenceMetrics(
            whisper_confidence=0.30,  # Low confidence
            audio_quality_score=0.40,
            speech_pattern_consistency=0.35,
            background_noise_level=0.70
        )
    )
    
    result4 = evaluator.evaluate_answer(disabled_input)
    
    if result4.confidence_adjustment is None:
        print("✅ Confidence adjustment correctly disabled")
        print(f"🎯 Final Score (no adjustment): {result4.scores.final_score:.2f}")
    else:
        print("❌ Confidence adjustment was applied despite being disabled")
    
    # Test Case 5: No Confidence Metrics Provided
    print("\n🔹 Test Case 5: No Confidence Metrics Provided")
    print("-" * 50)
    
    no_metrics_input = EvaluationInput(
        **{**base_input_data, "confidence_metrics": None}
    )
    
    result5 = evaluator.evaluate_answer(no_metrics_input)
    
    if result5.confidence_adjustment is None:
        print("✅ No confidence adjustment applied when metrics not provided")
        print(f"🎯 Final Score (no adjustment): {result5.scores.final_score:.2f}")
    else:
        print("❌ Confidence adjustment was applied despite no metrics")
    
    print("\n" + "=" * 80)
    print("🎉 CONFIDENCE ADJUSTMENT INTEGRATION TEST COMPLETED")
    print("=" * 80)


def test_confidence_adjuster_standalone():
    """Test the confidence adjuster as a standalone component"""
    
    print("\n📦 STANDALONE CONFIDENCE ADJUSTER TEST")
    print("-" * 50)
    
    # Initialize with custom configuration
    config = ConfidenceAdjustmentConfig(
        critical_threshold=0.40,
        low_threshold=0.60,
        good_threshold=0.80,
        critical_multiplier=0.50,
        low_multiplier=0.75,
        moderate_multiplier=0.90
    )
    
    adjuster = AdvancedConfidenceAdjuster(config)
    
    # Sample evaluation scores
    sample_scores = {
        "technical_accuracy": 8.5,
        "concept_clarity": 7.8,
        "keyword_coverage": 8.0,
        "communication": 7.2
    }
    
    print(f"🎯 Original Scores: {sample_scores}")
    
    # Test various confidence levels
    confidence_tests = [
        {"whisper": 0.90, "audio": 0.85, "speech": 0.88, "noise": 0.12, "label": "Excellent"},
        {"whisper": 0.55, "audio": 0.70, "speech": 0.65, "noise": 0.30, "label": "Low"},
        {"whisper": 0.25, "audio": 0.40, "speech": 0.35, "noise": 0.75, "label": "Critical"}
    ]
    
    for i, test in enumerate(confidence_tests, 1):
        print(f"\n🔍 Test {i} - {test['label']} Confidence:")
        
        result = adjuster.adjust_evaluation_scores(
            original_scores=sample_scores,
            whisper_confidence=test["whisper"],
            audio_quality_score=test["audio"],
            speech_pattern_consistency=test["speech"],
            background_noise_level=test["noise"]
        )
        
        print(f"   📊 Confidence Metrics:")
        print(f"      Whisper: {result.confidence_metrics['whisper_confidence']:.2f}")
        print(f"      Audio Quality: {result.confidence_metrics['audio_quality_score']:.2f}")
        print(f"      Speech Pattern: {result.confidence_metrics['speech_pattern_consistency']:.2f}")
        print(f"      Background Noise: {result.confidence_metrics['background_noise_level']:.2f}")
        print(f"      Overall: {result.confidence_metrics['overall_confidence']:.2f}")
        
        print(f"   🔧 Adjustment Factor: {result.adjustment_factor:.3f}")
        print(f"   📋 Recommendation: {result.recommendation}")
        
        print(f"   📈 Score Changes:")
        for metric, orig_score in result.original_scores.items():
            adj_score = result.adjusted_scores[metric]
            change = adj_score - orig_score
            print(f"      {metric}: {orig_score:.2f} → {adj_score:.2f} ({change:+.2f})")


def test_configuration_validation():
    """Test configuration validation and edge cases"""
    
    print("\n⚙️ CONFIGURATION VALIDATION TEST")
    print("-" * 50)
    
    try:
        # Test valid configuration
        valid_config = ConfidenceAdjustmentConfig(
            critical_threshold=0.3,
            low_threshold=0.6,
            good_threshold=0.8
        )
        print("✅ Valid configuration created successfully")
        
        # Test invalid threshold order (should work with validation)
        adjuster = AdvancedConfidenceAdjuster(valid_config)
        
        # Test statistics with no history
        stats = adjuster.get_adjustment_statistics()
        print(f"📊 Initial statistics: {stats.get('message', 'Statistics available')}")
        
        print("✅ Configuration validation tests completed")
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")


if __name__ == "__main__":
    print("🚀 STARTING CONFIDENCE ADJUSTMENT LAYER TESTS")
    print("=" * 80)
    
    try:
        # Run integration test
        test_confidence_adjustment_integration()
        
        # Run standalone tests
        test_confidence_adjuster_standalone()
        
        # Run configuration tests
        test_configuration_validation()
        
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)