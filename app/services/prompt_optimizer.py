# app/services/prompt_optimizer.py

import json
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.schemas import EvaluationInput, EvaluationOutput, ExperienceLevel, QuestionType
from app.services.llm_evaluator import LLMEvaluator
from app.utils.validator import validate_comprehensive_output, sanitize_llm_output

logger = logging.getLogger(__name__)

@dataclass
class PromptVersion:
    """Represents a specific prompt version for testing"""
    version_id: str
    name: str
    template: str
    description: str
    created_at: datetime
    test_results: Optional[Dict[str, Any]] = None

@dataclass
class PromptTestResult:
    """Results from testing a prompt version"""
    version_id: str
    test_case_id: str
    evaluation_result: Optional[EvaluationOutput]
    response_time_ms: float
    json_valid: bool
    hallucination_score: float  # 0-1, lower is better
    consistency_score: float   # 0-1, higher is better
    error_occurred: bool
    error_message: Optional[str] = None

@dataclass
class PromptComparisonReport:
    """Comprehensive comparison report between prompt versions"""
    comparison_id: str
    prompt_versions: List[str]
    test_cases_count: int
    results_summary: Dict[str, Dict[str, float]]
    best_version: str
    recommendations: List[str]
    detailed_results: List[PromptTestResult]
    generated_at: datetime


class PromptOptimizer:
    """Advanced Prompt Testing and Optimization Engine"""
    
    def __init__(self, evaluator: Optional[LLMEvaluator] = None):
        self.evaluator = evaluator or LLMEvaluator()
        self.prompt_versions: Dict[str, PromptVersion] = {}
        self.test_results: List[PromptTestResult] = []
        self.base_prompts_dir = Path(__file__).parent.parent / "prompts"
        
        # Load existing prompt versions
        self._load_prompt_versions()
        
    def _load_prompt_versions(self):
        """Load prompt versions from files and predefined variations"""
        
        # Load base prompt
        base_prompt_path = self.base_prompts_dir / "evaluation_prompt.txt"
        if base_prompt_path.exists():
            with open(base_prompt_path, "r", encoding="utf-8") as f:
                base_template = f.read()
                
            self.add_prompt_version(
                "base_v1",
                "Original Evaluation Prompt",
                base_template,
                "The original comprehensive evaluation prompt"
            )
        
        # Create optimized versions for testing
        self._create_optimization_variants()
    
    def _create_optimization_variants(self):
        """Create various prompt optimization variants for testing"""
        
        # Version 1: More concise prompt
        concise_prompt = """You are an expert AI interview evaluator. Use weighted 4-parameter scoring (0-10):

1. TECHNICAL ACCURACY (40%): Factual correctness, understanding, terminology
2. CONCEPT CLARITY (25%): Clear explanation, logical flow, appropriate depth  
3. KEYWORD COVERAGE (20%): Expected keywords: {expected_keywords}
4. COMMUNICATION (15%): Clarity, coherence, professional expression

ANTI-CHEATING: Detect copy-paste, AI-generated, robotic responses, transcript mismatch.

INPUT:
Question: {question}
Answer: {candidate_answer}
Level: {experience_level}
Type: {question_type}

Return ONLY JSON:
{{
  "scores": {{"technical_accuracy": <0-10>, "concept_clarity": <0-10>, "keyword_coverage": <0-10>, "communication": <0-10>, "final_score": <calculated>, "confidence_score": <1-10>}},
  "feedback": "<paragraph>",
  "anti_cheat": {{"is_copy_paste": <bool>, "is_ai_generated": <bool>, "is_too_robotic": <bool>, "transcript_mismatch": <bool>, "confidence_level": <0-1>, "risk_factors": [<list>]}},
  "keyword_analysis": {<keyword_mapping>},
  "response_quality": "<excellent/good/fair/poor>",
  "areas_for_improvement": [<list>],
  "processing_metadata": {{"answer_length": <int>, "complexity_level": "<low/medium/high>", "domain_match": <bool>}}
}}

Final score = (technical×0.4) + (clarity×0.25) + (keywords×0.2) + (comm×0.15)"""

        self.add_prompt_version(
            "concise_v1",
            "Concise Evaluation Prompt",
            concise_prompt,
            "Shorter, more focused version for better consistency"
        )
        
        # Version 2: Anti-hallucination focused
        anti_hallucination_prompt = """You are a precise AI interview evaluator. Follow these STRICT rules:

ONLY evaluate what is explicitly present in the candidate's answer.
DO NOT add assumptions or external knowledge.
Base scores ONLY on the provided answer content.

SCORING FRAMEWORK (0-10):
- Technical Accuracy (40%): Only what candidate stated, factual correctness
- Concept Clarity (25%): How clearly candidate explained (not what they didn't say)
- Keyword Coverage (20%): Which expected keywords were actually used: {expected_keywords}
- Communication (15%): Actual communication quality shown

EVALUATION DATA:
Question: {question}
Candidate Answer: {candidate_answer}
Experience Level: {experience_level}
Question Type: {question_type}

CRITICAL: Do not infer knowledge not demonstrated. Do not guess competence level beyond answer.

Return STRICT JSON format with exact structure shown in examples. NO additional text."""

        self.add_prompt_version(
            "anti_hallucination_v1",
            "Anti-Hallucination Focused Prompt",
            anti_hallucination_prompt,
            "Reduces hallucination by focusing only on explicit answer content"
        )
        
        # Version 3: Consistency enhanced
        consistency_prompt = """AI Interview Evaluator - Consistency Protocol

EVALUATION STANDARDS (Apply uniformly across all answers):

Technical Accuracy (40%):
- 8-10: Complete correct explanation with proper terminology
- 6-7: Mostly correct with minor gaps
- 4-5: Partially correct, some errors
- 2-3: Major errors, limited understanding
- 0-1: Incorrect or no technical content

Concept Clarity (25%):
- 8-10: Clear, well-structured explanation
- 6-7: Generally clear with some confusion
- 4-5: Some clarity issues, basic structure
- 2-3: Unclear explanation, poor structure
- 0-1: Very confusing or incomplete

Keyword Coverage (20%):
{expected_keywords}
- Score = (keywords_present / total_keywords) × 10

Communication (15%):
- Professional expression quality
- Appropriate for {experience_level} level

INPUT:
Question: {question}
Answer: {candidate_answer}

Apply scoring standards consistently. Return exact JSON format required."""

        self.add_prompt_version(
            "consistency_v1", 
            "Consistency Enhanced Prompt",
            consistency_prompt,
            "Emphasizes consistent scoring standards across evaluations"
        )
    
    def add_prompt_version(self, version_id: str, name: str, template: str, description: str):
        """Add a new prompt version for testing"""
        
        self.prompt_versions[version_id] = PromptVersion(
            version_id=version_id,
            name=name,
            template=template,
            description=description,
            created_at=datetime.utcnow()
        )
        logger.info(f"Added prompt version: {version_id} - {name}")
    
    def create_test_cases(self) -> List[EvaluationInput]:
        """Create comprehensive test cases for prompt evaluation"""
        
        test_cases = [
            # Test Case 1: Good Technical Answer
            EvaluationInput(
                question="Explain the difference between supervised and unsupervised learning.",
                candidate_answer="Supervised learning uses labeled training data where the algorithm learns from input-output pairs to make predictions on new data. Examples include classification and regression. Unsupervised learning finds patterns in data without labels, like clustering customers by behavior or dimensionality reduction for visualization.",
                expected_keywords=["supervised", "unsupervised", "labeled data", "classification", "regression", "clustering"],
                experience_level=ExperienceLevel.INTERMEDIATE,
                question_type=QuestionType.TECHNICAL
            ),
            
            # Test Case 2: Short, Incomplete Answer
            EvaluationInput(
                question="What is object-oriented programming?",
                candidate_answer="OOP uses classes and objects.",
                expected_keywords=["classes", "objects", "inheritance", "encapsulation", "polymorphism"],
                experience_level=ExperienceLevel.FRESHER,
                question_type=QuestionType.TECHNICAL
            ),
            
            # Test Case 3: Wrong Answer
            EvaluationInput(
                question="Explain database normalization.",
                candidate_answer="Database normalization is a process of encrypting sensitive data to prevent unauthorized access and ensure data security in cloud environments.",
                expected_keywords=["normalization", "redundancy", "tables", "relationships", "1NF", "2NF", "3NF"],
                experience_level=ExperienceLevel.ADVANCED,
                question_type=QuestionType.TECHNICAL
            ),
            
            # Test Case 4: Potential Copy-Paste
            EvaluationInput(
                question="What is machine learning?",
                candidate_answer="According to the textbook definition, machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns, and make decisions. Reference: Introduction to Statistical Learning by James et al.",
                expected_keywords=["machine learning", "algorithms", "data", "patterns", "artificial intelligence"],
                experience_level=ExperienceLevel.FRESHER,
                question_type=QuestionType.CONCEPTUAL
            ),
            
            # Test Case 5: Excellent Comprehensive Answer  
            EvaluationInput(
                question="Describe the SOLID principles in software engineering.",
                candidate_answer="SOLID principles are five design principles that make software designs more understandable, flexible, and maintainable. Single Responsibility Principle means each class should have one reason to change. Open-Closed Principle states classes should be open for extension but closed for modification. Liskov Substitution Principle ensures derived classes are substitutable for base classes. Interface Segregation Principle advocates for small, specific interfaces. Dependency Inversion Principle depends on abstractions, not concretions. These principles help create modular, testable code.",
                expected_keywords=["SOLID", "Single Responsibility", "Open-Closed", "Liskov Substitution", "Interface Segregation", "Dependency Inversion"],
                experience_level=ExperienceLevel.ADVANCED,
                question_type=QuestionType.TECHNICAL
            )
        ]
        
        return test_cases
    
    async def test_prompt_version(self, version_id: str, test_cases: List[EvaluationInput], 
                                repetitions: int = 3) -> List[PromptTestResult]:
        """Test a specific prompt version with multiple test cases"""
        
        if version_id not in self.prompt_versions:
            raise ValueError(f"Prompt version {version_id} not found")
        
        prompt_version = self.prompt_versions[version_id]
        results = []
        
        # Temporarily update evaluator's prompt template
        original_template = self.evaluator._get_current_template()  # We'll need to add this method
        self._update_evaluator_template(prompt_version.template)
        
        try:
            for i, test_case in enumerate(test_cases):
                test_case_id = f"test_case_{i+1}"
                
                # Run multiple times for consistency testing
                case_results = []
                for rep in range(repetitions):
                    start_time = time.time()
                    
                    try:
                        # Perform evaluation
                        evaluation_result = self.evaluator.evaluate_answer(test_case)
                        response_time = (time.time() - start_time) * 1000
                        
                        # Calculate hallucination score
                        hallucination_score = self._calculate_hallucination_score(
                            test_case, evaluation_result
                        )
                        
                        result = PromptTestResult(
                            version_id=version_id,
                            test_case_id=f"{test_case_id}_rep_{rep+1}",
                            evaluation_result=evaluation_result,
                            response_time_ms=response_time,
                            json_valid=True,
                            hallucination_score=hallucination_score,
                            consistency_score=0.0,  # Will calculate after all repetitions
                            error_occurred=False
                        )
                        
                    except Exception as e:
                        result = PromptTestResult(
                            version_id=version_id,
                            test_case_id=f"{test_case_id}_rep_{rep+1}",
                            evaluation_result=None,
                            response_time_ms=0.0,
                            json_valid=False,
                            hallucination_score=1.0,  # Max hallucination for errors
                            consistency_score=0.0,
                            error_occurred=True,
                            error_message=str(e)
                        )
                    
                    case_results.append(result)
                
                # Calculate consistency score for this test case
                if len(case_results) > 1:
                    consistency_score = self._calculate_consistency_score(case_results)
                    for result in case_results:
                        result.consistency_score = consistency_score
                
                results.extend(case_results)
                
        finally:
            # Restore original template
            self._update_evaluator_template(original_template)
        
        self.test_results.extend(results)
        return results
    
    def _update_evaluator_template(self, template: str):
        """Update the evaluator's prompt template"""
        # We'll need to modify the LLMEvaluator to support this
        # For now, this is a placeholder
        pass
    
    def _calculate_hallucination_score(self, test_case: EvaluationInput, 
                                     result: EvaluationOutput) -> float:
        """Calculate hallucination score (0-1, lower is better)"""
        
        hallucination_indicators = 0
        total_checks = 5
        
        # Check 1: Keyword analysis consistency
        answer_lower = test_case.candidate_answer.lower()
        for keyword, claimed_present in result.keyword_analysis.items():
            actual_present = keyword.lower() in answer_lower
            if claimed_present != actual_present:
                hallucination_indicators += 0.5
        
        # Check 2: Score reasonableness for answer length
        word_count = len(test_case.candidate_answer.split())
        if word_count < 10 and result.scores.final_score > 7:
            hallucination_indicators += 1
        
        # Check 3: Technical score vs. actual technical content
        technical_terms = sum(1 for keyword in test_case.expected_keywords 
                            if keyword.lower() in answer_lower)
        if technical_terms == 0 and result.scores.technical_accuracy > 6:
            hallucination_indicators += 1
        
        # Check 4: Feedback consistency with scores
        if result.scores.final_score > 8 and "excellent" not in result.feedback.lower():
            hallucination_indicators += 0.5
        
        # Check 5: Anti-cheat consistency
        if "textbook" in answer_lower and not result.anti_cheat.is_copy_paste:
            hallucination_indicators += 1
        
        return min(1.0, hallucination_indicators / total_checks)
    
    def _calculate_consistency_score(self, results: List[PromptTestResult]) -> float:
        """Calculate consistency score across multiple runs"""
        
        if len(results) < 2:
            return 1.0
        
        valid_results = [r for r in results if not r.error_occurred and r.evaluation_result]
        if len(valid_results) < 2:
            return 0.0
        
        # Calculate score variance
        scores = [r.evaluation_result.scores.final_score for r in valid_results]
        score_variance = statistics.variance(scores) if len(scores) > 1 else 0
        
        # Calculate response time variance  
        times = [r.response_time_ms for r in valid_results]
        time_variance = statistics.variance(times) if len(times) > 1 else 0
        
        # Consistency score (lower variance = higher consistency)
        score_consistency = max(0, 1 - (score_variance / 25))  # Normalize to 0-1
        time_consistency = max(0, 1 - (time_variance / 10000))  # Normalize to 0-1
        
        return (score_consistency + time_consistency) / 2
    
    async def compare_prompt_versions(self, version_ids: List[str], 
                                    test_cases: Optional[List[EvaluationInput]] = None) -> PromptComparisonReport:
        """Compare multiple prompt versions comprehensively"""
        
        if test_cases is None:
            test_cases = self.create_test_cases()
        
        comparison_results = {}
        all_results = []
        
        # Test each version
        for version_id in version_ids:
            logger.info(f"Testing prompt version: {version_id}")
            version_results = await self.test_prompt_version(version_id, test_cases)
            comparison_results[version_id] = version_results
            all_results.extend(version_results)
        
        # Calculate summary metrics
        summary = self._calculate_version_summaries(comparison_results)
        
        # Determine best version
        best_version = self._determine_best_version(summary)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(summary, comparison_results)
        
        report = PromptComparisonReport(
            comparison_id=f"comparison_{int(time.time())}",
            prompt_versions=version_ids,
            test_cases_count=len(test_cases),
            results_summary=summary,
            best_version=best_version,
            recommendations=recommendations,
            detailed_results=all_results,
            generated_at=datetime.utcnow()
        )
        
        # Save report
        self._save_comparison_report(report)
        
        return report
    
    def _calculate_version_summaries(self, results: Dict[str, List[PromptTestResult]]) -> Dict[str, Dict[str, float]]:
        """Calculate summary metrics for each version"""
        
        summary = {}
        
        for version_id, version_results in results.items():
            valid_results = [r for r in version_results if not r.error_occurred and r.evaluation_result]
            
            if not valid_results:
                summary[version_id] = {
                    "avg_score": 0.0,
                    "consistency": 0.0,
                    "hallucination": 1.0,
                    "response_time": 999999.0,
                    "json_success_rate": 0.0,
                    "error_rate": 1.0
                }
                continue
            
            # Calculate metrics
            avg_score = statistics.mean([r.evaluation_result.scores.final_score for r in valid_results])
            avg_consistency = statistics.mean([r.consistency_score for r in valid_results])
            avg_hallucination = statistics.mean([r.hallucination_score for r in valid_results])
            avg_response_time = statistics.mean([r.response_time_ms for r in valid_results])
            json_success_rate = len(valid_results) / len(version_results)
            error_rate = sum(1 for r in version_results if r.error_occurred) / len(version_results)
            
            summary[version_id] = {
                "avg_score": round(avg_score, 2),
                "consistency": round(avg_consistency, 3),
                "hallucination": round(avg_hallucination, 3),
                "response_time": round(avg_response_time, 1),
                "json_success_rate": round(json_success_rate, 3),
                "error_rate": round(error_rate, 3)
            }
        
        return summary
    
    def _determine_best_version(self, summary: Dict[str, Dict[str, float]]) -> str:
        """Determine the best prompt version based on weighted metrics"""
        
        weights = {
            "consistency": 0.3,
            "json_success_rate": 0.25,
            "hallucination": -0.2,  # Negative because lower is better
            "error_rate": -0.15,    # Negative because lower is better
            "response_time": -0.1   # Negative because lower is better
        }
        
        version_scores = {}
        
        for version_id, metrics in summary.items():
            score = 0
            for metric, weight in weights.items():
                value = metrics.get(metric, 0)
                if metric in ["hallucination", "error_rate"]:
                    value = 1 - value  # Invert for negative weights
                elif metric == "response_time":
                    value = max(0, 1 - (value / 10000))  # Normalize response time
                
                score += value * weight
            
            version_scores[version_id] = score
        
        return max(version_scores, key=version_scores.get)
    
    def _generate_recommendations(self, summary: Dict[str, Dict[str, float]], 
                                results: Dict[str, List[PromptTestResult]]) -> List[str]:
        """Generate optimization recommendations"""
        
        recommendations = []
        
        # Check for consistency issues
        low_consistency_versions = [v for v, s in summary.items() if s["consistency"] < 0.7]
        if low_consistency_versions:
            recommendations.append(f"Improve consistency in versions: {', '.join(low_consistency_versions)}. Consider more specific scoring criteria.")
        
        # Check for hallucination issues
        high_hallucination_versions = [v for v, s in summary.items() if s["hallucination"] > 0.3]
        if high_hallucination_versions:
            recommendations.append(f"Reduce hallucination in versions: {', '.join(high_hallucination_versions)}. Add stricter 'evaluate only what's present' instructions.")
        
        # Check for JSON issues
        low_json_versions = [v for v, s in summary.items() if s["json_success_rate"] < 0.9]
        if low_json_versions:
            recommendations.append(f"Improve JSON compliance in versions: {', '.join(low_json_versions)}. Enhance format enforcement.")
        
        # Check response times
        slow_versions = [v for v, s in summary.items() if s["response_time"] > 5000]
        if slow_versions:
            recommendations.append(f"Optimize response time for versions: {', '.join(slow_versions)}. Consider shorter prompts.")
        
        # Best practices
        if not recommendations:
            recommendations.append("All versions performing well. Consider A/B testing in production.")
        
        return recommendations
    
    def _save_comparison_report(self, report: PromptComparisonReport):
        """Save comparison report to file"""
        
        reports_dir = Path(__file__).parent.parent.parent / "reports" / "prompt_optimization"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = reports_dir / f"comparison_{report.comparison_id}.json"
        
        # Convert to JSON-serializable format
        report_data = {
            "comparison_id": report.comparison_id,
            "prompt_versions": report.prompt_versions,
            "test_cases_count": report.test_cases_count,
            "results_summary": report.results_summary,
            "best_version": report.best_version,
            "recommendations": report.recommendations,
            "generated_at": report.generated_at.isoformat(),
            "detailed_results": [
                {
                    "version_id": r.version_id,
                    "test_case_id": r.test_case_id,
                    "response_time_ms": r.response_time_ms,
                    "json_valid": r.json_valid,
                    "hallucination_score": r.hallucination_score,
                    "consistency_score": r.consistency_score,
                    "error_occurred": r.error_occurred,
                    "error_message": r.error_message,
                    "final_score": r.evaluation_result.scores.final_score if r.evaluation_result else None
                }
                for r in report.detailed_results
            ]
        }
        
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Comparison report saved: {report_file}")
    
    def generate_optimization_summary(self) -> Dict[str, Any]:
        """Generate overall optimization summary and insights"""
        
        if not self.test_results:
            return {"message": "No test results available. Run prompt comparisons first."}
        
        # Group results by version
        version_results = {}
        for result in self.test_results:
            if result.version_id not in version_results:
                version_results[result.version_id] = []
            version_results[result.version_id].append(result)
        
        # Calculate overall insights
        insights = {
            "total_tests_run": len(self.test_results),
            "versions_tested": len(version_results),
            "avg_hallucination_score": statistics.mean([r.hallucination_score for r in self.test_results]),
            "avg_consistency_score": statistics.mean([r.consistency_score for r in self.test_results]),
            "overall_json_success_rate": sum(1 for r in self.test_results if r.json_valid) / len(self.test_results),
            "version_performance": {}
        }
        
        # Add per-version insights
        for version_id, results in version_results.items():
            valid_results = [r for r in results if not r.error_occurred and r.evaluation_result]
            
            if valid_results:
                insights["version_performance"][version_id] = {
                    "avg_final_score": statistics.mean([r.evaluation_result.scores.final_score for r in valid_results]),
                    "consistency": statistics.mean([r.consistency_score for r in valid_results]),
                    "hallucination": statistics.mean([r.hallucination_score for r in valid_results]),
                    "success_rate": len(valid_results) / len(results)
                }
        
        return insights


# Utility function to run optimization easily
async def run_prompt_optimization(version_ids: Optional[List[str]] = None) -> PromptComparisonReport:
    """Convenient function to run prompt optimization"""
    
    optimizer = PromptOptimizer()
    
    if version_ids is None:
        version_ids = list(optimizer.prompt_versions.keys())
    
    logger.info(f"Starting prompt optimization with versions: {version_ids}")
    report = await optimizer.compare_prompt_versions(version_ids)
    
    # Print summary
    print("🎯 PROMPT OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Best Version: {report.best_version}")
    print(f"Test Cases: {report.test_cases_count}")
    print("\nRecommendations:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"{i}. {rec}")
    
    print(f"\nDetailed report saved with ID: {report.comparison_id}")
    
    return report