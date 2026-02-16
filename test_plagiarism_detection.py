"""
Test script for the Plagiarism Similarity Check integration

This script demonstrates the plagiarism detection functionality
and validates that it works correctly with the LLM evaluation system.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from app.schemas import EvaluationInput, ExperienceLevel, QuestionType, ConfidenceMetrics
    from app.services.llm_evaluator import LLMEvaluator
    from app.services.plagiarism_detector import AdvancedPlagiarismDetector, PlagiarismConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Note: Some dependencies may need to be installed:")
    print("pip install scikit-learn sentence-transformers spacy nltk")
    print("python -m spacy download en_core_web_sm")
    sys.exit(1)


def test_plagiarism_integration():
    """Test the complete integration of plagiarism detection with LLM evaluation"""
    
    print("=" * 80)
    print("🔍 PLAGIARISM SIMILARITY CHECK - INTEGRATION TEST")
    print("=" * 80)
    
    # Initialize the evaluator
    evaluator = LLMEvaluator()
    
    # Prepare test cases with different similarity levels
    question = "Explain the concept of microservices architecture and its benefits."
    expected_keywords = ["microservices", "architecture", "scalability", "independence", "deployment"]
    
    test_cases = [
        {
            "name": "Original Answer",
            "answer": """
            Microservices architecture is a design pattern where applications are built as a collection 
            of small, independent services that communicate through APIs. Each service handles a specific 
            business function and can be developed and deployed independently. The main benefits include 
            better scalability since you can scale individual services, improved fault isolation, 
            technology diversity as different services can use different tech stacks, and faster 
            development cycles through parallel team work.
            """,
            "expected_plagiarism": "none"
        },
        {
            "name": "Moderately Similar Answer",
            "answer": """
            Microservices represent an architectural approach where applications consist of loosely 
            coupled, independent services communicating via APIs. Each service focuses on a single 
            business capability and can be independently developed and deployed. Key advantages include 
            enhanced scalability through selective service scaling, better fault tolerance, technology 
            flexibility allowing different services to use various technologies, and accelerated 
            development through autonomous teams.
            """,
            "expected_plagiarism": "medium"
        },
        {
            "name": "Highly Similar Answer",
            "answer": """
            Microservices architecture is a design pattern where applications are constructed as a 
            collection of small, independent services that communicate through well-defined APIs. 
            Each individual service handles a specific business function and can be developed and 
            deployed independently of other services. The primary benefits include improved scalability 
            since you can scale individual services based on demand, enhanced fault isolation between 
            services, technology diversity as different services can utilize different technology stacks, 
            and faster development cycles through parallel team collaboration.
            """,
            "expected_plagiarism": "high"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['name']}")
        print("-" * 50)
        
        # Create evaluation input
        evaluation_input = EvaluationInput(
            question=question,
            candidate_answer=test_case["answer"],
            expected_keywords=expected_keywords,
            experience_level=ExperienceLevel.INTERMEDIATE,
            question_type=QuestionType.TECHNICAL,
            enable_plagiarism_detection=True,
            enable_confidence_adjustment=False  # Focus on plagiarism for this test
        )
        
        # Perform evaluation
        try:
            result = evaluator.evaluate_answer(evaluation_input)
            
            # Display basic anti-cheat results
            print(f"📊 Anti-Cheat Results:")
            print(f"   Plagiarism Risk: {result.anti_cheat.plagiarism_risk or 'Not detected'}")
            print(f"   Similarity Score: {result.anti_cheat.similarity_score:.1%}" if result.anti_cheat.similarity_score else "   Similarity Score: Not calculated")
            print(f"   Detection Confidence: {result.anti_cheat.plagiarism_confidence:.1%}" if result.anti_cheat.plagiarism_confidence else "   Detection Confidence: Not available")
            
            # Display detailed plagiarism analysis if available
            if result.plagiarism_analysis:
                print(f"\n📋 Detailed Plagiarism Analysis:")
                print(f"   Risk Level: {result.plagiarism_analysis.risk_level}")
                print(f"   Overall Similarity: {result.plagiarism_analysis.overall_similarity:.1%}")
                print(f"   Analysis Confidence: {result.plagiarism_analysis.confidence:.1%}")
                print(f"   Explanation: {result.plagiarism_analysis.explanation}")
                print(f"   Recommendation: {result.plagiarism_analysis.recommendation}")
                
                # Show similarity breakdown
                print(f"\n📊 Similarity Breakdown:")
                for method, score in result.plagiarism_analysis.similarity_breakdown.items():
                    print(f"      {method.replace('_', ' ').title()}: {score:.1%}")
                
                # Show flagged sections if any
                if result.plagiarism_analysis.flagged_sections:
                    print(f"\n🚨 Flagged Sections ({len(result.plagiarism_analysis.flagged_sections)}):")
                    for j, section in enumerate(result.plagiarism_analysis.flagged_sections[:2]):  # Show first 2
                        print(f"      Section {j+1}: {section['similarity']:.1%} similar")
                        print(f"         Candidate: {section['candidate_sentence'][:80]}...")
                        print(f"         Ideal: {section['ideal_sentence'][:80]}...")
                
                print(f"\n⏱️  Processing Time: {result.plagiarism_analysis.processing_time_ms:.1f}ms")
            else:
                print(f"\n❌ No detailed plagiarism analysis available")
            
            # Show impact on overall evaluation
            print(f"\n🎯 Overall Evaluation Impact:")
            print(f"   Final Score: {result.scores.final_score:.2f}")
            print(f"   Anti-Cheat Risk Factors: {len(result.anti_cheat.risk_factors)}")
            if result.anti_cheat.risk_factors:
                for factor in result.anti_cheat.risk_factors[:3]:  # Show first 3
                    print(f"      - {factor}")
            
        except Exception as e:
            print(f"❌ Test case failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Test with plagiarism detection disabled
    print(f"\n🔹 Test Case: Plagiarism Detection Disabled")
    print("-" * 50)
    
    disabled_input = EvaluationInput(
        question=question,
        candidate_answer=test_cases[2]["answer"],  # Use highly similar answer
        expected_keywords=expected_keywords,
        experience_level=ExperienceLevel.INTERMEDIATE,
        question_type=QuestionType.TECHNICAL,
        enable_plagiarism_detection=False
    )
    
    try:
        result = evaluator.evaluate_answer(disabled_input)
        
        if (result.anti_cheat.plagiarism_risk is None and 
            result.plagiarism_analysis is None):
            print("✅ Plagiarism detection correctly disabled")
        else:
            print("❌ Plagiarism detection was not properly disabled")
        
        print(f"🎯 Final Score (no plagiarism check): {result.scores.final_score:.2f}")
        
    except Exception as e:
        print(f"❌ Disabled test failed: {e}")


def test_plagiarism_detector_standalone():
    """Test the plagiarism detector as a standalone component"""
    
    print("\n📦 STANDALONE PLAGIARISM DETECTOR TEST")
    print("-" * 50)
    
    try:
        # Initialize detector with custom config
        config = PlagiarismConfig(
            critical_threshold=0.85,
            high_threshold=0.70,
            medium_threshold=0.55,
            low_threshold=0.40
        )
        
        detector = AdvancedPlagiarismDetector(config)
        
        # Test question and answers
        question = "What are the advantages of using Docker containers?"
        expected_keywords = ["Docker", "containers", "virtualization", "deployment", "isolation"]
        
        test_answers = [
            {
                "name": "Short Answer",
                "answer": "Docker containers provide isolation and easy deployment.",
                "expectation": "Too short for analysis"
            },
            {
                "name": "Original Answer", 
                "answer": """
                Docker containers offer several key advantages for modern application deployment.
                First, they provide lightweight virtualization that uses fewer resources than traditional VMs.
                Second, containers ensure consistent environments across development, testing, and production.
                Third, they enable rapid scaling and deployment of applications.
                """,
                "expectation": "Low similarity"
            }
        ]
        
        for i, test in enumerate(test_answers, 1):
            print(f"\n🔍 Test {i} - {test['name']}:")
            
            try:
                result = detector.analyze_plagiarism(
                    candidate_answer=test["answer"],
                    question=question,
                    expected_keywords=expected_keywords,
                    experience_level="intermediate"
                )
                
                print(f"   📊 Risk Level: {result.risk_level}")
                print(f"   📈 Overall Similarity: {result.overall_similarity:.1%}")
                print(f"   🎯 Confidence: {result.confidence:.1%}")
                print(f"   💡 Explanation: {result.explanation}")
                print(f"   📋 Recommendation: {result.recommendation}")
                print(f"   ⏱️  Processing Time: {result.processing_time_ms:.1f}ms")
                
                # Validate expectation
                print(f"   ✅ Expectation: {test['expectation']}")
                
            except Exception as e:
                print(f"   ❌ Analysis failed: {e}")
        
    except Exception as e:
        print(f"❌ Standalone test setup failed: {e}")
        import traceback
        traceback.print_exc()


def test_configuration_and_caching():
    """Test configuration options and caching functionality"""
    
    print("\n⚙️ CONFIGURATION AND CACHING TEST")
    print("-" * 50)
    
    try:
        detector = AdvancedPlagiarismDetector()
        
        question = "Explain database normalization."
        keywords = ["normalization", "database", "tables", "redundancy"]
        
        # Test ideal answer generation and caching
        print("🧪 Testing ideal answer generation:")
        
        start_time = time.time()
        ideal1 = detector.generate_ideal_answer(question, keywords, "intermediate", use_cache=True)
        time1 = time.time() - start_time
        print(f"   First generation: {time1*1000:.1f}ms")
        
        start_time = time.time()
        ideal2 = detector.generate_ideal_answer(question, keywords, "intermediate", use_cache=True)
        time2 = time.time() - start_time
        print(f"   Second generation (cached): {time2*1000:.1f}ms")
        
        if ideal1 == ideal2 and time2 < time1:
            print("   ✅ Caching working correctly")
        else:
            print("   ⚠️  Caching may not be working as expected")
        
        print(f"   📝 Generated ideal answer length: {len(ideal1.split())} words")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")


if __name__ == "__main__":
    print("🚀 STARTING PLAGIARISM SIMILARITY CHECK TESTS")
    print("=" * 80)
    
    # Import time for timing tests
    import time
    
    try:
        # Run integration test
        test_plagiarism_integration()
        
        # Run standalone tests
        test_plagiarism_detector_standalone()
        
        # Run configuration tests
        test_configuration_and_caching()
        
        print("\n🎉 ALL PLAGIARISM DETECTION TESTS COMPLETED!")
        
    except Exception as e:
        print(f"\n❌ TESTS FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)