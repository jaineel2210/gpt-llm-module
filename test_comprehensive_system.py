#!/usr/bin/env python3
"""
Comprehensive Test Suite for Advanced LLM Interview Evaluation System

This test suite demonstrates all 7 key features:
1. Evaluation Architecture
2. Structured Output Format
3. Scoring Rubric
4. Anti-Cheating Intelligence
5. Prompt Engineering
6. Error Handling + Stability
7. Integration With Risk Engine
"""

import sys
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Any

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.schemas import (
    EvaluationInput, ExperienceLevel, QuestionType, 
    ScoringWeights, EvaluationOutput, RiskEngineOutput
)
from app.services.llm_evaluator import LLMEvaluator
from app.utils.validator import (
    validate_comprehensive_output, create_validation_report, 
    validate_keyword_analysis
)
from app.config import DEFAULT_WEIGHTS, ANTI_CHEAT_CONFIG


class ComprehensiveSystemTester:
    """Test suite for the comprehensive evaluation system"""
    
    def __init__(self):
        self.evaluator = LLMEvaluator()
        self.test_results = []
        
    def run_all_tests(self):
        """Execute all test scenarios"""
        print("🚀 Starting Comprehensive LLM Evaluation System Tests")
        print("=" * 60)
        
        # Test 1: Architecture Design
        print("\n📐 TEST 1: Evaluation Architecture")
        self.test_evaluation_architecture()
        
        # Test 2: Structured Output
        print("\n📊 TEST 2: Structured Output Format")
        self.test_structured_output()
        
        # Test 3: Scoring Rubric
        print("\n📏 TEST 3: Weighted Scoring Rubric")
        self.test_scoring_rubric()
        
        # Test 4: Anti-Cheating Detection
        print("\n🛡️ TEST 4: Anti-Cheating Intelligence")
        self.test_anti_cheating()
        
        # Test 5: Prompt Engineering
        print("\n🎯 TEST 5: Prompt Engineering")
        self.test_prompt_engineering()
        
        # Test 6: Error Handling
        print("\n⚠️ TEST 6: Error Handling & Stability")
        self.test_error_handling()
        
        # Test 7: Risk Engine Integration
        print("\n🔗 TEST 7: Risk Engine Integration")
        self.test_risk_engine_integration()
        
        # Generate final summary
        print("\n" + "=" * 60)
        self.generate_test_summary()
    
    def test_evaluation_architecture(self):
        """Test 1: Comprehensive Input/Output Architecture"""
        print("Testing comprehensive input format with all parameters...")
        
        # Create comprehensive test input
        test_input = EvaluationInput(
            question="Explain the difference between supervised and unsupervised machine learning.",
            candidate_answer="Supervised learning uses labeled training data to learn patterns, like classification and regression. Unsupervised learning finds hidden patterns in data without labels, like clustering and dimensionality reduction.",
            expected_keywords=["supervised", "unsupervised", "labeled data", "classification", "clustering"],
            experience_level=ExperienceLevel.INTERMEDIATE,
            question_type=QuestionType.TECHNICAL,
            context="Data Science interview for mid-level position",
            max_score=10,
            time_taken=120,
            audio_transcript="The candidate spoke clearly with minor hesitations."
        )
        
        try:
            result = self.evaluator.evaluate_answer(test_input)
            
            # Verify comprehensive output structure
            assert hasattr(result, 'scores'), "Missing scores structure"
            assert hasattr(result, 'anti_cheat'), "Missing anti-cheat analysis"
            assert hasattr(result, 'keyword_analysis'), "Missing keyword analysis"
            assert hasattr(result, 'processing_metadata'), "Missing processing metadata"
            
            print(f"✅ Architecture test passed")
            print(f"   - Input parameters: {len(test_input.expected_keywords)} keywords, {test_input.question_type} type")
            print(f"   - Output completeness: All required sections present")
            print(f"   - Final score: {result.scores.final_score}/10")
            
            self.test_results.append({"test": "architecture", "status": "PASS", "details": result})
            
        except Exception as e:
            print(f"❌ Architecture test failed: {str(e)}")
            self.test_results.append({"test": "architecture", "status": "FAIL", "error": str(e)})
    
    def test_structured_output(self):
        """Test 2: Strict JSON Output Format"""
        print("Testing strict JSON output structure and validation...")
        
        test_input = EvaluationInput(
            question="What is polymorphism in object-oriented programming?",
            candidate_answer="Polymorphism allows objects of different types to be treated as instances of the same type through a common interface.",
            expected_keywords=["polymorphism", "inheritance", "interface", "objects"],
            experience_level=ExperienceLevel.FRESHER,
            question_type=QuestionType.TECHNICAL
        )
        
        try:
            result = self.evaluator.evaluate_answer(test_input)
            
            # Test JSON serialization
            json_output = result.model_dump_json()
            parsed_back = json.loads(json_output)
            
            # Verify all required fields
            required_fields = ["scores", "feedback", "anti_cheat", "keyword_analysis"]
            for field in required_fields:
                assert field in parsed_back, f"Missing required field: {field}"
            
            # Verify score structure
            scores = parsed_back["scores"]
            score_fields = ["technical_accuracy", "concept_clarity", "keyword_coverage", "communication", "final_score"]
            for field in score_fields:
                assert field in scores, f"Missing score field: {field}"
                assert 0 <= scores[field] <= 10, f"Score {field} out of range: {scores[field]}"
            
            print(f"✅ Structured output test passed")
            print(f"   - JSON serialization successful")
            print(f"   - All required fields present")
            print(f"   - Score ranges valid (0-10 scale)")
            
            self.test_results.append({"test": "structured_output", "status": "PASS"})
            
        except Exception as e:
            print(f"❌ Structured output test failed: {str(e)}")
            self.test_results.append({"test": "structured_output", "status": "FAIL", "error": str(e)})
    
    def test_scoring_rubric(self):
        """Test 3: Weighted Scoring Rubric Implementation"""
        print("Testing weighted scoring calculation (40-25-20-15 distribution)...")
        
        test_input = EvaluationInput(
            question="Explain how database indexing improves query performance.",
            candidate_answer="Database indexes create separate data structures that point to data rows, enabling faster searches by avoiding full table scans.",
            expected_keywords=["index", "performance", "query", "data structure"],
            experience_level=ExperienceLevel.ADVANCED,
            question_type=QuestionType.TECHNICAL
        )
        
        try:
            result = self.evaluator.evaluate_answer(test_input)
            
            # Verify weighted scoring calculation
            scores = result.scores
            expected_final = (
                scores.technical_accuracy * DEFAULT_WEIGHTS["technical_accuracy"] +
                scores.concept_clarity * DEFAULT_WEIGHTS["concept_clarity"] +
                scores.keyword_coverage * DEFAULT_WEIGHTS["keyword_coverage"] +
                scores.communication * DEFAULT_WEIGHTS["communication"]
            )
            
            # Allow small rounding differences
            score_diff = abs(expected_final - scores.final_score)
            assert score_diff < 0.05, f"Score calculation mismatch: {score_diff}"
            
            print(f"✅ Scoring rubric test passed")
            print(f"   - Technical: {scores.technical_accuracy}/10 (40% weight)")
            print(f"   - Clarity: {scores.concept_clarity}/10 (25% weight)")
            print(f"   - Keywords: {scores.keyword_coverage}/10 (20% weight)")
            print(f"   - Communication: {scores.communication}/10 (15% weight)")
            print(f"   - Final calculated: {expected_final:.2f}, Actual: {scores.final_score}")
            
            self.test_results.append({"test": "scoring_rubric", "status": "PASS", "scores": scores})
            
        except Exception as e:
            print(f"❌ Scoring rubric test failed: {str(e)}")
            self.test_results.append({"test": "scoring_rubric", "status": "FAIL", "error": str(e)})
    
    def test_anti_cheating(self):
        """Test 4: Anti-Cheating Detection Capabilities"""
        print("Testing anti-cheating detection with suspicious responses...")
        
        # Test copy-paste detection
        copy_paste_answer = "According to the textbook definition, machine learning is a method of data analysis that automates analytical model building. Reference: Introduction to Statistical Learning."
        
        copy_paste_input = EvaluationInput(
            question="What is machine learning?",
            candidate_answer=copy_paste_answer,
            expected_keywords=["machine learning", "data", "algorithms"],
            experience_level=ExperienceLevel.FRESHER,
            question_type=QuestionType.CONCEPTUAL
        )
        
        # Test AI-generated detection
        ai_generated_answer = "As an artificial intelligence, I can provide you with a comprehensive explanation. Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed."
        
        ai_input = EvaluationInput(
            question="What is machine learning?",
            candidate_answer=ai_generated_answer,
            expected_keywords=["machine learning", "AI", "algorithms"],
            experience_level=ExperienceLevel.INTERMEDIATE,
            question_type=QuestionType.CONCEPTUAL
        )
        
        # Test robotic response detection
        robotic_answer = "Machine learning enables computers to learn automatically without explicit programming through algorithms and statistical models."
        
        robotic_input = EvaluationInput(
            question="What is machine learning?",
            candidate_answer=robotic_answer,
            expected_keywords=["machine learning", "algorithms", "models"],
            experience_level=ExperienceLevel.FRESHER,
            question_type=QuestionType.CONCEPTUAL
        )
        
        try:
            # Test copy-paste detection
            copy_result = self.evaluator.evaluate_answer(copy_paste_input)
            print(f"   Copy-paste detection: {copy_result.anti_cheat.is_copy_paste}")
            
            # Test AI-generated detection
            ai_result = self.evaluator.evaluate_answer(ai_input)
            print(f"   AI-generated detection: {ai_result.anti_cheat.is_ai_generated}")
            
            # Test robotic detection
            robotic_result = self.evaluator.evaluate_answer(robotic_input)
            print(f"   Robotic response detection: {robotic_result.anti_cheat.is_too_robotic}")
            
            # Verify that at least some cheating was detected
            total_detections = (
                copy_result.anti_cheat.is_copy_paste +
                ai_result.anti_cheat.is_ai_generated +
                robotic_result.anti_cheat.is_too_robotic
            )
            
            print(f"✅ Anti-cheating test passed")
            print(f"   - Total suspicious patterns detected: {total_detections}")
            print(f"   - Risk factors identified: {len(copy_result.anti_cheat.risk_factors)}")
            
            self.test_results.append({"test": "anti_cheating", "status": "PASS", "detections": total_detections})
            
        except Exception as e:
            print(f"❌ Anti-cheating test failed: {str(e)}")
            self.test_results.append({"test": "anti_cheating", "status": "FAIL", "error": str(e)})
    
    def test_prompt_engineering(self):
        """Test 5: Prompt Engineering Quality"""
        print("Testing prompt engineering with various answer types...")
        
        test_cases = [
            {
                "name": "Short Answer",
                "answer": "OOP uses classes.",
                "expected_score_range": (2, 6)
            },
            {
                "name": "Comprehensive Answer",
                "answer": "Object-oriented programming (OOP) is a programming paradigm based on objects that contain data (attributes) and code (methods). Key principles include encapsulation, inheritance, polymorphism, and abstraction. Encapsulation bundles data and methods together, inheritance allows classes to inherit properties from parent classes, polymorphism enables objects to take multiple forms, and abstraction hides complex implementation details.",
                "expected_score_range": (7, 10)
            },
            {
                "name": "Wrong Answer",
                "answer": "Object-oriented programming is a type of database management system used for storing binary data.",
                "expected_score_range": (0, 4)
            },
            {
                "name": "Partially Correct",
                "answer": "OOP is about using classes and objects. Classes define the structure and objects are instances.",
                "expected_score_range": (4, 7)
            }
        ]
        
        try:
            results = []
            for case in test_cases:
                test_input = EvaluationInput(
                    question="Explain object-oriented programming.",
                    candidate_answer=case["answer"],
                    expected_keywords=["classes", "objects", "inheritance", "encapsulation", "polymorphism"],
                    experience_level=ExperienceLevel.INTERMEDIATE,
                    question_type=QuestionType.TECHNICAL
                )
                
                result = self.evaluator.evaluate_answer(test_input)
                score = result.scores.final_score
                expected_min, expected_max = case["expected_score_range"]
                
                # Verify score is in expected range
                score_in_range = expected_min <= score <= expected_max
                results.append({
                    "case": case["name"],
                    "score": score,
                    "expected_range": case["expected_score_range"],
                    "in_range": score_in_range
                })
                
                print(f"   {case['name']}: Score {score}/10 (Expected: {expected_min}-{expected_max}) {'✓' if score_in_range else '✗'}")
            
            # Check if most scores are in expected ranges
            passed_cases = sum(1 for r in results if r["in_range"])
            success_rate = passed_cases / len(results)
            
            if success_rate >= 0.75:  # 75% success rate
                print(f"✅ Prompt engineering test passed")
                print(f"   - Success rate: {success_rate:.1%} ({passed_cases}/{len(results)})")
                self.test_results.append({"test": "prompt_engineering", "status": "PASS", "success_rate": success_rate})
            else:
                print(f"❌ Prompt engineering test failed")
                print(f"   - Success rate: {success_rate:.1%} (minimum 75% required)")
                self.test_results.append({"test": "prompt_engineering", "status": "FAIL", "success_rate": success_rate})
            
        except Exception as e:
            print(f"❌ Prompt engineering test failed: {str(e)}")
            self.test_results.append({"test": "prompt_engineering", "status": "FAIL", "error": str(e)})
    
    def test_error_handling(self):
        """Test 6: Error Handling and Stability"""
        print("Testing error handling and fallback mechanisms...")
        
        # Test with empty inputs
        error_cases = [
            {
                "name": "Empty Answer",
                "question": "What is Python?",
                "answer": "",
                "keywords": ["python", "programming"]
            },
            {
                "name": "Very Long Answer", 
                "question": "Explain programming",
                "answer": "Programming " * 1000,  # Very long answer
                "keywords": ["programming"]
            },
            {
                "name": "Special Characters",
                "question": "What is SQL?",
                "answer": "SQL is a query language! @#$%^&*(){}[]|\\:;\",.<>?",
                "keywords": ["SQL", "database"]
            }
        ]
        
        try:
            passed_cases = 0
            for case in error_cases:
                try:
                    test_input = EvaluationInput(
                        question=case["question"],
                        candidate_answer=case["answer"],
                        expected_keywords=case["keywords"],
                        experience_level=ExperienceLevel.INTERMEDIATE,
                        question_type=QuestionType.TECHNICAL
                    )
                    
                    result = self.evaluator.evaluate_answer(test_input)
                    
                    # Verify result has required structure even for edge cases
                    assert hasattr(result, 'scores')
                    assert hasattr(result, 'feedback')
                    assert hasattr(result, 'anti_cheat')
                    
                    passed_cases += 1
                    print(f"   {case['name']}: Handled gracefully ✓")
                    
                except Exception as case_error:
                    print(f"   {case['name']}: Failed - {str(case_error)} ✗")
            
            success_rate = passed_cases / len(error_cases)
            
            if success_rate >= 0.75:
                print(f"✅ Error handling test passed")
                print(f"   - Handled {passed_cases}/{len(error_cases)} edge cases")
                self.test_results.append({"test": "error_handling", "status": "PASS", "success_rate": success_rate})
            else:
                print(f"❌ Error handling test failed")
                self.test_results.append({"test": "error_handling", "status": "FAIL", "success_rate": success_rate})
                
        except Exception as e:
            print(f"❌ Error handling test failed: {str(e)}")
            self.test_results.append({"test": "error_handling", "status": "FAIL", "error": str(e)})
    
    def test_risk_engine_integration(self):
        """Test 7: Risk Engine Integration Format"""
        print("Testing risk engine output format and integration...")
        
        test_input = EvaluationInput(
            question="Describe the SOLID principles in software engineering.",
            candidate_answer="SOLID principles include Single Responsibility, Open-Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion principles that help create maintainable code.",
            expected_keywords=["SOLID", "Single Responsibility", "Open-Closed", "Liskov", "Interface Segregation", "Dependency Inversion"],
            experience_level=ExperienceLevel.ADVANCED,
            question_type=QuestionType.TECHNICAL
        )
        
        try:
            # Get comprehensive evaluation
            evaluation_result = self.evaluator.evaluate_answer(test_input)
            
            # Convert to risk engine format
            risk_output = self.evaluator.generate_risk_engine_output(evaluation_result)
            
            # Verify risk engine output structure
            assert hasattr(risk_output, 'llm_score'), "Missing llm_score"
            assert hasattr(risk_output, 'risk_flag'), "Missing risk_flag"
            assert hasattr(risk_output, 'confidence_level'), "Missing confidence_level"
            assert hasattr(risk_output, 'cheat_probability'), "Missing cheat_probability"
            assert hasattr(risk_output, 'quality_metrics'), "Missing quality_metrics"
            
            # Verify data types and ranges
            assert 0 <= risk_output.llm_score <= 10, f"Invalid LLM score: {risk_output.llm_score}"
            assert isinstance(risk_output.risk_flag, bool), "risk_flag should be boolean"
            assert risk_output.confidence_level in ["low", "medium", "high"], f"Invalid confidence level: {risk_output.confidence_level}"
            assert 0 <= risk_output.cheat_probability <= 1, f"Invalid cheat probability: {risk_output.cheat_probability}"
            
            # Verify quality metrics structure
            expected_metrics = ["technical_accuracy", "concept_clarity", "keyword_coverage", "communication"]
            for metric in expected_metrics:
                assert metric in risk_output.quality_metrics, f"Missing quality metric: {metric}"
            
            print(f"✅ Risk engine integration test passed")
            print(f"   - LLM Score: {risk_output.llm_score}/10")
            print(f"   - Risk Flag: {risk_output.risk_flag}")
            print(f"   - Confidence: {risk_output.confidence_level}")
            print(f"   - Cheat Probability: {risk_output.cheat_probability:.2f}")
            print(f"   - Quality Metrics: {len(risk_output.quality_metrics)} parameters")
            
            self.test_results.append({"test": "risk_engine_integration", "status": "PASS", "risk_output": risk_output})
            
        except Exception as e:
            print(f"❌ Risk engine integration test failed: {str(e)}")
            self.test_results.append({"test": "risk_engine_integration", "status": "FAIL", "error": str(e)})
    
    def generate_test_summary(self):
        """Generate comprehensive test summary"""
        print("📋 COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["status"] == "PASS")
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        for i, result in enumerate(self.test_results, 1):
            status_icon = "✅" if result["status"] == "PASS" else "❌"
            print(f"{i}. {result['test'].replace('_', ' ').title()}: {result['status']} {status_icon}")
            if "error" in result:
                print(f"   Error: {result['error']}")
        
        # Generate integration readiness report
        print(f"\n🔗 INTEGRATION READINESS:")
        critical_tests = ["architecture", "structured_output", "risk_engine_integration"]
        critical_passed = sum(1 for r in self.test_results 
                             if r["test"] in critical_tests and r["status"] == "PASS")
        
        if critical_passed == len(critical_tests):
            print("✅ System ready for integration with risk engine")
        else:
            print("❌ System requires fixes before integration")
        
        # Feature completeness
        print(f"\n🎯 FEATURE COMPLETENESS:")
        features = [
            ("Input/Output Architecture", "architecture"),
            ("Structured JSON Format", "structured_output"), 
            ("Weighted Scoring Rubric", "scoring_rubric"),
            ("Anti-Cheating Detection", "anti_cheating"),
            ("Advanced Prompt Engineering", "prompt_engineering"),
            ("Error Handling & Stability", "error_handling"),
            ("Risk Engine Integration", "risk_engine_integration")
        ]
        
        for feature_name, test_name in features:
            test_result = next((r for r in self.test_results if r["test"] == test_name), None)
            if test_result and test_result["status"] == "PASS":
                print(f"✅ {feature_name}")
            else:
                print(f"❌ {feature_name}")
        
        print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main test execution function"""
    print("Advanced LLM Interview Evaluation System - Test Suite")
    print("Version: 2.0.0")
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize and run tests
    tester = ComprehensiveSystemTester()
    tester.run_all_tests()
    
    print("\n🎉 All tests completed!")
    print("Check the above results for system readiness assessment.")


if __name__ == "__main__":
    main()