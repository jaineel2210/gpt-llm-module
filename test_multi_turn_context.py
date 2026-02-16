"""
Test script for the Multi-Turn Interview Context integration

This script demonstrates the multi-turn context functionality
and validates that it works correctly with the LLM evaluation system.
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.schemas import EvaluationInput, ExperienceLevel, QuestionType
from app.services.llm_evaluator import LLMEvaluator
from app.services.context_manager import MultiTurnContextManager, ContextConfig


def test_multi_turn_integration():
    """Test the complete integration of multi-turn context with LLM evaluation"""
    
    print("=" * 80)
    print("🔄 MULTI-TURN INTERVIEW CONTEXT - INTEGRATION TEST")
    print("=" * 80)
    
    # Initialize the evaluator
    evaluator = LLMEvaluator()
    
    # Prepare a simulated interview sequence
    interview_id = "INT_MULTITEST_20241218"
    candidate_id = "CANDIDATE_123"
    
    # Define an interview sequence that tests different aspects
    interview_questions = [
        {
            "question": "What is Object-Oriented Programming and why is it important?",
            "answer": """
            Object-Oriented Programming (OOP) is a programming paradigm based on the concept of objects 
            that contain both data and methods. The core principles include encapsulation, inheritance, 
            and polymorphism. It's important because it promotes code reusability, maintainability, 
            and helps organize complex software into manageable components.
            """,
            "expected_keywords": ["OOP", "objects", "encapsulation", "inheritance", "polymorphism", "reusability"],
            "question_type": QuestionType.TECHNICAL,
            "test_aspect": "Initial technical question"
        },
        {
            "question": "Can you explain inheritance in more detail and give an example?",
            "answer": """
            Inheritance is one of the fundamental principles of OOP I mentioned earlier. It allows 
            a class to inherit properties and methods from a parent class. For example, you might 
            have a Vehicle class with properties like speed and methods like start(). Then you could 
            create a Car class that inherits from Vehicle but adds specific features like number of doors.
            """,
            "expected_keywords": ["inheritance", "class", "parent", "properties", "methods", "example"],
            "question_type": QuestionType.TECHNICAL,
            "test_aspect": "Follow-up question referencing previous answer"
        },
        {
            "question": "How do you handle challenging deadlines in software projects?",
            "answer": """
            When facing tight deadlines, I prioritize tasks based on business impact and dependencies. 
            I communicate early and often with stakeholders about realistic timelines. I focus on 
            delivering a minimum viable solution first, then iterate. I also leverage my technical 
            knowledge, like the OOP principles we discussed, to write more maintainable code that's 
            easier to debug and extend.
            """,
            "expected_keywords": ["deadlines", "prioritize", "communication", "stakeholders", "MVP"],
            "question_type": QuestionType.BEHAVIORAL,
            "test_aspect": "Behavioral question with reference to technical discussion"
        },
        {
            "question": "What's the difference between composition and inheritance in OOP?",
            "answer": """
            Great follow-up to our inheritance discussion! Composition and inheritance are both ways 
            to achieve code reuse, but they work differently. Inheritance creates an 'is-a' relationship, 
            like a Car 'is-a' Vehicle. Composition creates a 'has-a' relationship, where objects contain 
            other objects. Composition is often preferred because it's more flexible and avoids issues 
            like the diamond problem in multiple inheritance.
            """,
            "expected_keywords": ["composition", "inheritance", "is-a", "has-a", "code reuse", "flexible"],
            "question_type": QuestionType.TECHNICAL,
            "test_aspect": "Advanced technical question building on previous answers"
        }
    ]
    
    print(f"🎬 Starting interview simulation: {interview_id}")
    print(f"👤 Candidate: {candidate_id}")
    print(f"📝 Questions to ask: {len(interview_questions)}")
    
    # Process each question in sequence
    for i, qa in enumerate(interview_questions, 1):
        print(f"\n🔹 Question {i}/{len(interview_questions)}: {qa['test_aspect']}")
        print("-" * 60)
        
        # Create evaluation input
        evaluation_input = EvaluationInput(
            question=qa["question"],
            candidate_answer=qa["answer"],
            expected_keywords=qa["expected_keywords"],
            experience_level=ExperienceLevel.INTERMEDIATE,
            question_type=qa["question_type"],
            interview_id=interview_id,
            candidate_id=candidate_id,
            enable_multi_turn_context=True,
            enable_confidence_adjustment=False,  # Focus on context for this test
            enable_plagiarism_detection=False   # Focus on context for this test
        )
        
        # Perform evaluation
        try:
            start_time = time.time()
            result = evaluator.evaluate_answer(evaluation_input)
            evaluation_time = time.time() - start_time
            
            # Display basic results
            print(f"📊 Evaluation Results:")
            print(f"   Final Score: {result.scores.final_score:.2f}")
            print(f"   Technical Accuracy: {result.scores.technical_accuracy:.2f}")
            print(f"   Concept Clarity: {result.scores.concept_clarity:.2f}")
            print(f"   Keywords Found: {sum(result.keyword_analysis.values())}/{len(qa['expected_keywords'])}")
            
            # Display context analysis results
            if result.context_analysis:
                print(f"\n🔄 Context Analysis:")
                print(f"   Total Questions So Far: {result.context_analysis.total_questions}")
                print(f"   Consistency Level: {result.context_analysis.consistency_level or 'Not yet available'}")
                print(f"   Consistency Score: {result.context_analysis.consistency_score:.2f}" if result.context_analysis.consistency_score else "   Consistency Score: Not yet calculated")
                print(f"   Topics Covered: {', '.join(result.context_analysis.topic_progression[-5:])}")  # Last 5 topics
                print(f"   Context Summary: {result.context_analysis.context_summary}")
                print(f"   Context Influence: {result.context_analysis.context_influence}")
            else:
                print(f"\n❌ No context analysis available")
            
            # Display any relevant feedback about context
            if "context" in result.feedback.lower() or "previous" in result.feedback.lower():
                print(f"\n💬 Context-Related Feedback:")
                context_sentences = [s for s in result.feedback.split('.') if 'context' in s.lower() or 'previous' in s.lower()]
                for sentence in context_sentences[:2]:  # Show up to 2 relevant sentences
                    print(f"   - {sentence.strip()}")
            
            print(f"\n⏱️  Processing Time: {evaluation_time*1000:.1f}ms")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Test context manager statistics
    print(f"\n📈 FINAL INTERVIEW STATISTICS")
    print("-" * 60)
    
    try:
        # Access context manager directly to get detailed statistics
        context_manager = evaluator.context_manager
        stats = context_manager.get_context_statistics(interview_id)
        
        print(f"📊 Context Statistics:")
        print(f"   Total Questions: {stats['total_questions']}")
        print(f"   Interview Duration: {stats['duration_minutes']:.1f} minutes")
        print(f"   Average Score: {stats['average_score']:.2f}")
        print(f"   Score Range: {stats['score_range'].get('min', 0):.1f} - {stats['score_range'].get('max', 0):.1f}")
        print(f"   Question Types: {', '.join(stats['question_types'])}")
        print(f"   Topics Covered: {stats['topics_covered']}")
        print(f"   Summary: {stats['context_summary']}")
        
        if 'consistency' in stats:
            print(f"\n🎯 Consistency Analysis:")
            consistency = stats['consistency']
            print(f"   Consistency Level: {consistency['level']}")
            print(f"   Consistency Score: {consistency['score']:.2f}")
            print(f"   Inconsistencies Found: {consistency['inconsistencies']}")
        
    except Exception as e:
        print(f"❌ Failed to get context statistics: {e}")


def test_multi_turn_disabled():
    """Test evaluation with multi-turn context disabled"""
    
    print(f"\n🔹 TEST: Multi-Turn Context Disabled")
    print("-" * 50)
    
    evaluator = LLMEvaluator()
    
    evaluation_input = EvaluationInput(
        question="What is machine learning?",
        candidate_answer="Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
        expected_keywords=["machine learning", "AI", "data", "programming"],
        experience_level=ExperienceLevel.INTERMEDIATE,
        question_type=QuestionType.TECHNICAL,
        interview_id="DISABLED_TEST_123",
        enable_multi_turn_context=False  # Explicitly disabled
    )
    
    try:
        result = evaluator.evaluate_answer(evaluation_input)
        
        if result.context_analysis is None:
            print("✅ Multi-turn context correctly disabled")
        else:
            print("❌ Multi-turn context was not properly disabled")
        
        print(f"🎯 Final Score (no context): {result.scores.final_score:.2f}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


def test_context_manager_standalone():
    """Test the context manager as a standalone component"""
    
    print(f"\n📦 STANDALONE CONTEXT MANAGER TEST")
    print("-" * 50)
    
    # Initialize with custom config
    config = ContextConfig(
        max_context_pairs=3,
        context_window_minutes=30,
        enable_consistency_analysis=True,
        storage_type="memory"
    )
    
    context_manager = MultiTurnContextManager(config)
    
    interview_id = "STANDALONE_TEST_456"
    
    # Simulate adding multiple Q&A pairs
    test_pairs = [
        {"question": "What is Python?", "answer": "Python is a high-level programming language.", 
         "type": "technical", "score": 8.0, "keywords": ["Python", "programming", "language"]},
        {"question": "Explain Python lists", "answer": "Lists are ordered collections in Python that can store multiple items.", 
         "type": "technical", "score": 7.5, "keywords": ["lists", "collections", "items"]},
        {"question": "What's your Python experience?", "answer": "I've been using Python for 3 years in various projects.", 
         "type": "behavioral", "score": 6.0, "keywords": ["experience", "projects"]}
    ]
    
    print(f"🧪 Adding {len(test_pairs)} Q&A pairs:")
    
    for i, pair in enumerate(test_pairs, 1):
        try:
            context = context_manager.add_question_answer(
                interview_id=interview_id,
                question=pair["question"],
                answer=pair["answer"],
                question_type=pair["type"],
                score=pair["score"],
                keywords_covered=pair["keywords"]
            )
            
            print(f"   Q{i}: Added successfully - Total questions: {context.total_questions}")
            
        except Exception as e:
            print(f"   Q{i}: Failed to add - {e}")
    
    # Test context retrieval
    print(f"\n📋 Context for Evaluation:")
    context_str = context_manager.get_context_for_evaluation(interview_id)
    if context_str:
        print(f"   Context Length: {len(context_str)} characters")
        print(f"   First 200 chars: {context_str[:200]}...")
    else:
        print("   No context available")
    
    # Test statistics
    print(f"\n📊 Standalone Statistics:")
    stats = context_manager.get_context_statistics(interview_id)
    print(f"   Average Score: {stats.get('average_score', 0):.2f}")
    print(f"   Question Types: {stats.get('question_types', [])}")
    print(f"   Topics: {stats.get('topics_covered', 0)}")


if __name__ == "__main__":
    print("🚀 STARTING MULTI-TURN CONTEXT TESTS")
    print("=" * 80)
    
    try:
        # Run integration test
        test_multi_turn_integration()
        
        # Test with context disabled
        test_multi_turn_disabled()
        
        # Run standalone tests
        test_context_manager_standalone()
        
        print("\n🎉 ALL MULTI-TURN CONTEXT TESTS COMPLETED!")
        
    except Exception as e:
        print(f"\n❌ TESTS FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)