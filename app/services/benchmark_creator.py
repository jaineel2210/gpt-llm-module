# app/services/benchmark_creator.py

import json
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from app.schemas import EvaluationInput, ExperienceLevel, QuestionType
from app.services.llm_evaluator import LLMEvaluator

logger = logging.getLogger(__name__)

class AnswerQuality(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    BAD = "bad"

@dataclass
class BenchmarkAnswer:
    """Represents a benchmark answer for testing"""
    id: str
    question: str
    answer: str
    expected_keywords: List[str]
    experience_level: ExperienceLevel
    question_type: QuestionType
    quality_level: AnswerQuality
    expected_score_range: Tuple[float, float]
    rationale: str
    context: Optional[str] = None
    
@dataclass
class BenchmarkEvaluationResult:
    """Result of evaluating a benchmark answer"""
    benchmark_id: str
    quality_level: AnswerQuality
    expected_range: Tuple[float, float]
    actual_score: float
    within_range: bool
    score_accuracy: float  # How close to expected range
    evaluation_time_ms: float
    detailed_scores: Dict[str, float]
    feedback: str
    anti_cheat_flags: Dict[str, bool]
    
@dataclass 
class BenchmarkReport:
    """Comprehensive benchmark evaluation report"""
    dataset_id: str
    total_cases: int
    quality_distribution: Dict[str, int]
    overall_accuracy: float
    range_accuracy_by_quality: Dict[str, float]
    avg_evaluation_time: float
    score_distribution: Dict[str, List[float]]
    detailed_results: List[BenchmarkEvaluationResult]
    fairness_metrics: Dict[str, Any]
    generated_at: datetime


class BenchmarkDatasetCreator:
    """Creates comprehensive benchmark datasets for LLM evaluation testing"""
    
    def __init__(self):
        self.datasets_dir = Path(__file__).parent.parent.parent / "benchmark_datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
    def create_comprehensive_dataset(self) -> List[BenchmarkAnswer]:
        """Create a comprehensive benchmark dataset with varied quality levels"""
        
        dataset = []
        
        # Technical Questions
        dataset.extend(self._create_technical_answers())
        
        # Conceptual Questions
        dataset.extend(self._create_conceptual_answers())
        
        # System Design Questions
        dataset.extend(self._create_system_design_answers())
        
        # Behavioral Questions
        dataset.extend(self._create_behavioral_answers())
        
        # Edge Cases
        dataset.extend(self._create_edge_case_answers())
        
        # Shuffle for randomness
        random.shuffle(dataset)
        
        logger.info(f"Created comprehensive dataset with {len(dataset)} benchmark answers")
        return dataset
    
    def _create_technical_answers(self) -> List[BenchmarkAnswer]:
        """Create technical question benchmark answers"""
        
        answers = []
        
        # Machine Learning Question
        ml_question = "Explain the difference between supervised and unsupervised learning with examples."
        ml_keywords = ["supervised", "unsupervised", "labeled data", "training", "classification", "regression", "clustering"]
        
        # Excellent Answer
        answers.append(BenchmarkAnswer(
            id="tech_ml_excellent",
            question=ml_question,
            answer="Supervised learning uses labeled training data where we have input-output pairs to train algorithms. The model learns to map inputs to correct outputs. Examples include classification (email spam detection) and regression (house price prediction). Unsupervised learning finds patterns in unlabeled data without target variables. Examples include clustering (customer segmentation using k-means), dimensionality reduction (PCA for visualization), and association rules (market basket analysis). The key difference is that supervised learning has ground truth labels for training while unsupervised learning discovers hidden structures in data.",
            expected_keywords=ml_keywords,
            experience_level=ExperienceLevel.ADVANCED,
            question_type=QuestionType.TECHNICAL,
            quality_level=AnswerQuality.EXCELLENT,
            expected_score_range=(8.5, 10.0),
            rationale="Comprehensive explanation with clear definitions, multiple examples, and key distinctions"
        ))
        
        # Good Answer
        answers.append(BenchmarkAnswer(
            id="tech_ml_good",
            question=ml_question,
            answer="Supervised learning uses labeled data to train models. It includes classification and regression. For example, predicting house prices from features like size and location. Unsupervised learning works with unlabeled data to find patterns. Clustering is a common example where we group similar customers together. The main difference is supervised learning has target labels while unsupervised doesn't.",
            expected_keywords=ml_keywords,
            experience_level=ExperienceLevel.INTERMEDIATE,
            question_type=QuestionType.TECHNICAL,
            quality_level=AnswerQuality.GOOD,
            expected_score_range=(7.0, 8.4),
            rationale="Solid understanding with examples but less comprehensive than excellent"
        ))
        
        # Average Answer
        answers.append(BenchmarkAnswer(
            id="tech_ml_average",
            question=ml_question,
            answer="Supervised learning has labels and unsupervised learning doesn't. In supervised learning you train with examples and answers. Unsupervised learning finds patterns by itself. Classification and clustering are examples of these types.",
            expected_keywords=ml_keywords,
            experience_level=ExperienceLevel.FRESHER,
            question_type=QuestionType.TECHNICAL,
            quality_level=AnswerQuality.AVERAGE,
            expected_score_range=(5.0, 6.9),
            rationale="Basic understanding but lacks depth and specific examples"
        ))
        
        # Poor Answer  
        answers.append(BenchmarkAnswer(
            id="tech_ml_poor",
            question=ml_question,
            answer="Supervised learning is when you supervise the computer while it learns. Unsupervised learning is when the computer learns by itself without supervision. They are both types of AI.",
            expected_keywords=ml_keywords,
            experience_level=ExperienceLevel.FRESHER,
            question_type=QuestionType.TECHNICAL,
            quality_level=AnswerQuality.POOR,
            expected_score_range=(2.0, 4.9),
            rationale="Fundamental misunderstanding of concepts"
        ))
        
        # Bad Answer
        answers.append(BenchmarkAnswer(
            id="tech_ml_bad",
            question=ml_question,
            answer="Machine learning is about machines that learn things. I don't really know the difference between the two types you mentioned.",
            expected_keywords=ml_keywords,
            experience_level=ExperienceLevel.FRESHER,
            question_type=QuestionType.TECHNICAL,
            quality_level=AnswerQuality.BAD,
            expected_score_range=(0.0, 1.9),
            rationale="No understanding demonstrated, admits ignorance"
        ))
        
        # OOP Question
        oop_question = "Explain the four main principles of object-oriented programming."
        oop_keywords = ["encapsulation", "inheritance", "polymorphism", "abstraction", "classes", "objects"]
        
        # Excellent Answer
        answers.append(BenchmarkAnswer(
            id="tech_oop_excellent",
            question=oop_question,
            answer="The four main principles of OOP are: 1) Encapsulation - bundling data and methods together in classes, providing data hiding and access control through private/public modifiers. 2) Inheritance - creating new classes based on existing ones, enabling code reuse and is-a relationships. 3) Polymorphism - objects taking multiple forms, allowing same interface for different types through method overriding and overloading. 4) Abstraction - hiding complex implementation details behind simple interfaces, focusing on what an object does rather than how.",
            expected_keywords=oop_keywords,
            experience_level=ExperienceLevel.ADVANCED,
            question_type=QuestionType.TECHNICAL,
            quality_level=AnswerQuality.EXCELLENT,
            expected_score_range=(8.5, 10.0),
            rationale="All four principles explained clearly with implementation details"
        ))
        
        # Good Answer
        answers.append(BenchmarkAnswer(
            id="tech_oop_good",
            question=oop_question,
            answer="The four principles are encapsulation, inheritance, polymorphism, and abstraction. Encapsulation keeps data and methods together in classes. Inheritance lets classes inherit from other classes. Polymorphism means objects can take different forms. Abstraction hides complexity from users.",
            expected_keywords=oop_keywords,
            experience_level=ExperienceLevel.INTERMEDIATE,
            question_type=QuestionType.TECHNICAL,
            quality_level=AnswerQuality.GOOD,
            expected_score_range=(7.0, 8.4),
            rationale="All principles mentioned with basic explanations"
        ))
        
        # Average Answer
        answers.append(BenchmarkAnswer(
            id="tech_oop_average",
            question=oop_question,
            answer="OOP has encapsulation, inheritance, and polymorphism. Encapsulation is about hiding data. Inheritance is when classes inherit from other classes. Polymorphism is having multiple forms.",
            expected_keywords=oop_keywords,
            experience_level=ExperienceLevel.INTERMEDIATE,
            question_type=QuestionType.TECHNICAL,
            quality_level=AnswerQuality.AVERAGE,
            expected_score_range=(5.0, 6.9),
            rationale="Missing one principle (abstraction), basic explanations"
        ))
        
        # Poor Answer
        answers.append(BenchmarkAnswer(
            id="tech_oop_poor",
            question=oop_question,
            answer="Object-oriented programming uses objects and classes. The main principles are about organizing code in objects.",
            expected_keywords=oop_keywords,
            experience_level=ExperienceLevel.FRESHER,
            question_type=QuestionType.TECHNICAL,
            quality_level=AnswerQuality.POOR,
            expected_score_range=(2.0, 4.9),
            rationale="Vague answer, doesn't name specific principles"
        ))
        
        return answers
    
    def _create_conceptual_answers(self) -> List[BenchmarkAnswer]:
        """Create conceptual question benchmark answers"""
        
        answers = []
        
        # Database normalization question
        db_question = "What is database normalization and why is it important?"
        db_keywords = ["normalization", "redundancy", "1NF", "2NF", "3NF", "tables", "relationships", "anomalies"]
        
        # Excellent Answer
        answers.append(BenchmarkAnswer(
            id="concept_db_excellent",
            question=db_question,
            answer="Database normalization is the process of organizing data to reduce redundancy and improve data integrity. It involves decomposing tables into smaller, well-structured tables and defining relationships between them. The main normal forms are 1NF (eliminating repeating groups), 2NF (removing partial dependencies), and 3NF (eliminating transitive dependencies). Normalization prevents update, insert, and delete anomalies, ensures data consistency, reduces storage space, and makes the database easier to maintain. However, it may require more complex queries and joins.",
            expected_keywords=db_keywords,
            experience_level=ExperienceLevel.ADVANCED,
            question_type=QuestionType.CONCEPTUAL,
            quality_level=AnswerQuality.EXCELLENT,
            expected_score_range=(8.5, 10.0),
            rationale="Comprehensive explanation including process, normal forms, benefits, and trade-offs"
        ))
        
        # Average Answer
        answers.append(BenchmarkAnswer(
            id="concept_db_average",
            question=db_question,
            answer="Database normalization removes duplicate data from databases. It makes tables more organized and reduces redundancy. This helps prevent errors when updating data.",
            expected_keywords=db_keywords,
            experience_level=ExperienceLevel.INTERMEDIATE,
            question_type=QuestionType.CONCEPTUAL,
            quality_level=AnswerQuality.AVERAGE,
            expected_score_range=(5.0, 6.9),
            rationale="Basic understanding but lacks detail about normal forms and specific benefits"
        ))
        
        # Bad Answer
        answers.append(BenchmarkAnswer(
            id="concept_db_bad",
            question=db_question,
            answer="Normalization is making databases normal. It's important because databases should be normal.",
            expected_keywords=db_keywords,
            experience_level=ExperienceLevel.FRESHER,
            question_type=QuestionType.CONCEPTUAL,
            quality_level=AnswerQuality.BAD,
            expected_score_range=(0.0, 1.9),
            rationale="Circular definition showing no real understanding"
        ))
        
        return answers
    
    def _create_system_design_answers(self) -> List[BenchmarkAnswer]:
        """Create system design benchmark answers"""
        
        answers = []
        
        # Scalability question
        scale_question = "How would you design a system to handle millions of users?"
        scale_keywords = ["scalability", "load balancing", "database sharding", "caching", "CDN", "microservices", "horizontal scaling"]
        
        # Excellent Answer
        answers.append(BenchmarkAnswer(
            id="system_scale_excellent",
            question=scale_question,
            answer="To handle millions of users, I'd implement horizontal scaling with load balancers distributing traffic across multiple application servers. Use database sharding to partition data across multiple database instances. Implement multiple caching layers: Redis for session data, CDN for static content, and application-level caching for frequently accessed data. Adopt microservices architecture to scale individual components independently. Use message queues for asynchronous processing. Monitor with centralized logging and implement circuit breakers for fault tolerance. Consider eventual consistency where appropriate and use auto-scaling groups in cloud environments.",
            expected_keywords=scale_keywords,
            experience_level=ExperienceLevel.ADVANCED,
            question_type=QuestionType.SYSTEM_DESIGN,
            quality_level=AnswerQuality.EXCELLENT,
            expected_score_range=(8.5, 10.0),
            rationale="Comprehensive system design covering multiple scalability techniques"
        ))
        
        # Good Answer  
        answers.append(BenchmarkAnswer(
            id="system_scale_good",
            question=scale_question,
            answer="I would use load balancers to distribute traffic across multiple servers. Implement caching to reduce database load. Use database replication and consider sharding for very large datasets. Deploy the application across multiple data centers for better geographic distribution.",
            expected_keywords=scale_keywords,
            experience_level=ExperienceLevel.INTERMEDIATE,
            question_type=QuestionType.SYSTEM_DESIGN,
            quality_level=AnswerQuality.GOOD,
            expected_score_range=(7.0, 8.4),
            rationale="Good understanding of key concepts but less comprehensive"
        ))
        
        # Poor Answer
        answers.append(BenchmarkAnswer(
            id="system_scale_poor",
            question=scale_question,
            answer="Use a bigger server with more RAM and CPU. Maybe add more servers if needed.",
            expected_keywords=scale_keywords,
            experience_level=ExperienceLevel.FRESHER,
            question_type=QuestionType.SYSTEM_DESIGN,
            quality_level=AnswerQuality.POOR,
            expected_score_range=(2.0, 4.9),
            rationale="Shows vertical scaling thinking but lacks understanding of distributed systems"
        ))
        
        return answers
    
    def _create_behavioral_answers(self) -> List[BenchmarkAnswer]:
        """Create behavioral question benchmark answers"""
        
        answers = []
        
        # Problem-solving question
        problem_question = "Describe a challenging technical problem you solved and your approach."
        problem_keywords = ["problem", "solution", "approach", "debugging", "analysis", "implementation", "testing"]
        
        # Excellent Answer
        answers.append(BenchmarkAnswer(
            id="behavioral_problem_excellent",
            question=problem_question,
            answer="I encountered a performance issue where our API response times increased from 200ms to 3+ seconds after a deployment. I systematically debugged by first reproducing the issue locally, then analyzing application logs and database query performance. I discovered that a new feature introduced N+1 query problems. I implemented eager loading with proper JOIN queries, added database indexes on frequently queried columns, and introduced query result caching. I also set up monitoring alerts to catch similar issues early. The fix reduced response times to 150ms and improved overall system reliability.",
            expected_keywords=problem_keywords,
            experience_level=ExperienceLevel.ADVANCED,
            question_type=QuestionType.BEHAVIORAL,
            quality_level=AnswerQuality.EXCELLENT,
            expected_score_range=(8.5, 10.0),
            rationale="Structured approach with clear problem identification, root cause analysis, solution, and outcome"
        ))
        
        # Average Answer
        answers.append(BenchmarkAnswer(
            id="behavioral_problem_average",
            question=problem_question,
            answer="I had a bug in my code that was causing crashes. I looked at the logs and found the error. I fixed the code by adding error handling and the problem was solved.",
            expected_keywords=problem_keywords,
            experience_level=ExperienceLevel.INTERMEDIATE,
            question_type=QuestionType.BEHAVIORAL,
            quality_level=AnswerQuality.AVERAGE,
            expected_score_range=(5.0, 6.9),
            rationale="Basic problem-solving approach but lacks detail and systematic methodology"
        ))
        
        return answers
    
    def _create_edge_case_answers(self) -> List[BenchmarkAnswer]:
        """Create edge case answers for testing robustness"""
        
        answers = []
        
        # Very short answer
        answers.append(BenchmarkAnswer(
            id="edge_short",
            question="Explain RESTful APIs.",
            answer="REST uses HTTP.",
            expected_keywords=["REST", "HTTP", "resources", "stateless", "CRUD", "JSON"],
            experience_level=ExperienceLevel.FRESHER,
            question_type=QuestionType.TECHNICAL,
            quality_level=AnswerQuality.BAD,
            expected_score_range=(0.0, 2.9),
            rationale="Extremely short answer with minimal information"
        ))
        
        # Very long answer (potential copy-paste)
        answers.append(BenchmarkAnswer(
            id="edge_long_copypaste", 
            question="What is machine learning?",
            answer="According to the textbook definition, machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence (AI) based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks. Reference: Introduction to Statistical Learning, James et al., 2013.",
            expected_keywords=["machine learning", "algorithms", "data", "patterns", "AI"],
            experience_level=ExperienceLevel.FRESHER,
            question_type=QuestionType.CONCEPTUAL,
            quality_level=AnswerQuality.POOR,
            expected_score_range=(1.0, 4.0),
            rationale="Obvious copy-paste from textbook with citation"
        ))
        
        # Wrong but confident answer
        answers.append(BenchmarkAnswer(
            id="edge_wrong_confident",
            question="Explain the difference between SQL and NoSQL databases.",
            answer="SQL databases are newer and faster than NoSQL databases. SQL stands for 'Structured Query Language' which means they can only store structured data like spreadsheets. NoSQL databases are the old traditional databases that can only store unstructured data. Companies like Google and Facebook use SQL databases because they're more modern and scalable.",
            expected_keywords=["SQL", "NoSQL", "relational", "schema", "ACID", "scalability", "flexibility"],
            experience_level=ExperienceLevel.INTERMEDIATE,
            question_type=QuestionType.TECHNICAL,
            quality_level=AnswerQuality.BAD,
            expected_score_range=(0.0, 2.0),
            rationale="Confident but completely incorrect information"
        ))
        
        return answers
    
    def save_dataset(self, dataset: List[BenchmarkAnswer], dataset_name: str = "comprehensive") -> Path:
        """Save benchmark dataset to file"""
        
        dataset_data = {
            "dataset_name": dataset_name,
            "created_at": datetime.utcnow().isoformat(),
            "total_answers": len(dataset),
            "quality_distribution": {
                quality.value: len([a for a in dataset if a.quality_level == quality])
                for quality in AnswerQuality
            },
            "answers": [
                {
                    "id": answer.id,
                    "question": answer.question,
                    "answer": answer.answer,
                    "expected_keywords": answer.expected_keywords,
                    "experience_level": answer.experience_level.value,
                    "question_type": answer.question_type.value,
                    "quality_level": answer.quality_level.value,
                    "expected_score_range": answer.expected_score_range,
                    "rationale": answer.rationale,
                    "context": answer.context
                }
                for answer in dataset
            ]
        }
        
        file_path = self.datasets_dir / f"{dataset_name}_dataset.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dataset_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved benchmark dataset to: {file_path}")
        return file_path
    
    def load_dataset(self, dataset_name: str) -> Optional[List[BenchmarkAnswer]]:
        """Load benchmark dataset from file"""
        
        file_path = self.datasets_dir / f"{dataset_name}_dataset.json"
        if not file_path.exists():
            logger.error(f"Dataset file not found: {file_path}")
            return None
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            dataset = []
            for answer_data in data["answers"]:
                answer = BenchmarkAnswer(
                    id=answer_data["id"],
                    question=answer_data["question"],
                    answer=answer_data["answer"],
                    expected_keywords=answer_data["expected_keywords"],
                    experience_level=ExperienceLevel(answer_data["experience_level"]),
                    question_type=QuestionType(answer_data["question_type"]),
                    quality_level=AnswerQuality(answer_data["quality_level"]),
                    expected_score_range=tuple(answer_data["expected_score_range"]),
                    rationale=answer_data["rationale"],
                    context=answer_data.get("context")
                )
                dataset.append(answer)
            
            logger.info(f"Loaded {len(dataset)} benchmark answers from {file_path}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return None


class BenchmarkEvaluator:
    """Evaluates LLM performance using benchmark datasets"""
    
    def __init__(self, evaluator: Optional[LLMEvaluator] = None):
        self.evaluator = evaluator or LLMEvaluator()
        self.results_dir = Path(__file__).parent.parent.parent / "benchmark_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_dataset(self, dataset: List[BenchmarkAnswer], 
                        report_name: str = "benchmark_evaluation") -> BenchmarkReport:
        """Evaluate LLM performance on benchmark dataset"""
        
        logger.info(f"Starting benchmark evaluation on {len(dataset)} answers")
        
        results = []
        quality_distribution = {}
        score_distribution = {quality.value: [] for quality in AnswerQuality}
        total_time = 0
        
        for benchmark in dataset:
            # Convert to evaluation input
            eval_input = EvaluationInput(
                question=benchmark.question,
                candidate_answer=benchmark.answer,
                expected_keywords=benchmark.expected_keywords,
                experience_level=benchmark.experience_level,
                question_type=benchmark.question_type,
                context=benchmark.context
            )
            
            # Perform evaluation
            start_time = time.time()
            try:
                evaluation_result = self.evaluator.evaluate_answer(eval_input)
                evaluation_time = (time.time() - start_time) * 1000
                total_time += evaluation_time
                
                # Check if score is within expected range
                actual_score = evaluation_result.scores.final_score
                expected_min, expected_max = benchmark.expected_score_range
                within_range = expected_min <= actual_score <= expected_max
                
                # Calculate score accuracy (how close to expected range)
                if within_range:
                    score_accuracy = 1.0
                else:
                    # Distance from nearest boundary
                    distance = min(abs(actual_score - expected_min), abs(actual_score - expected_max))
                    score_accuracy = max(0, 1 - distance / 5)  # Normalize by half range
                
                result = BenchmarkEvaluationResult(
                    benchmark_id=benchmark.id,
                    quality_level=benchmark.quality_level,
                    expected_range=benchmark.expected_score_range,
                    actual_score=actual_score,
                    within_range=within_range,
                    score_accuracy=score_accuracy,
                    evaluation_time_ms=evaluation_time,
                    detailed_scores={
                        "technical_accuracy": evaluation_result.scores.technical_accuracy,
                        "concept_clarity": evaluation_result.scores.concept_clarity,
                        "keyword_coverage": evaluation_result.scores.keyword_coverage,
                        "communication": evaluation_result.scores.communication
                    },
                    feedback=evaluation_result.feedback,
                    anti_cheat_flags={
                        "is_copy_paste": evaluation_result.anti_cheat.is_copy_paste,
                        "is_ai_generated": evaluation_result.anti_cheat.is_ai_generated,
                        "is_too_robotic": evaluation_result.anti_cheat.is_too_robotic,
                        "transcript_mismatch": evaluation_result.anti_cheat.transcript_mismatch
                    }
                )
                
                results.append(result)
                score_distribution[benchmark.quality_level.value].append(actual_score)
                
                # Update quality distribution
                quality = benchmark.quality_level.value
                quality_distribution[quality] = quality_distribution.get(quality, 0) + 1
                
            except Exception as e:
                logger.error(f"Evaluation failed for {benchmark.id}: {e}")
                # Create failure result
                result = BenchmarkEvaluationResult(
                    benchmark_id=benchmark.id,
                    quality_level=benchmark.quality_level,
                    expected_range=benchmark.expected_score_range,
                    actual_score=0.0,
                    within_range=False,
                    score_accuracy=0.0,
                    evaluation_time_ms=0.0,
                    detailed_scores={},
                    feedback=f"Evaluation failed: {str(e)}",
                    anti_cheat_flags={}
                )
                results.append(result)
        
        # Calculate metrics
        successful_results = [r for r in results if r.actual_score > 0]
        overall_accuracy = sum(r.score_accuracy for r in successful_results) / len(successful_results) if successful_results else 0
        
        range_accuracy_by_quality = {}
        for quality in AnswerQuality:
            quality_results = [r for r in successful_results if r.quality_level == quality]
            if quality_results:
                range_accuracy_by_quality[quality.value] = sum(r.within_range for r in quality_results) / len(quality_results)
        
        avg_evaluation_time = total_time / len(results) if results else 0
        
        # Fairness metrics
        fairness_metrics = self._calculate_fairness_metrics(results, dataset)
        
        # Create report
        report = BenchmarkReport(
            dataset_id=f"{report_name}_{int(time.time())}",
            total_cases=len(dataset),
            quality_distribution=quality_distribution,
            overall_accuracy=overall_accuracy,
            range_accuracy_by_quality=range_accuracy_by_quality,
            avg_evaluation_time=avg_evaluation_time,
            score_distribution=score_distribution,
            detailed_results=results,
            fairness_metrics=fairness_metrics,
            generated_at=datetime.utcnow()
        )
        
        # Save report
        self._save_report(report)
        
        logger.info(f"Benchmark evaluation completed. Overall accuracy: {overall_accuracy:.2%}")
        return report
    
    def _calculate_fairness_metrics(self, results: List[BenchmarkEvaluationResult], 
                                   dataset: List[BenchmarkAnswer]) -> Dict[str, Any]:
        """Calculate fairness metrics across different dimensions"""
        
        metrics = {}
        
        # Score variance by experience level
        experience_scores = {}
        for result, benchmark in zip(results, dataset):
            level = benchmark.experience_level.value
            if level not in experience_scores:
                experience_scores[level] = []
            experience_scores[level].append(result.actual_score)
        
        # Calculate coefficient of variation for each experience level
        experience_fairness = {}
        for level, scores in experience_scores.items():
            if len(scores) > 1:
                mean_score = sum(scores) / len(scores)
                variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
                std_dev = variance ** 0.5
                cv = std_dev / mean_score if mean_score > 0 else 0
                experience_fairness[level] = {
                    "mean_score": mean_score,
                    "coefficient_variation": cv,
                    "sample_count": len(scores)
                }
        
        metrics["experience_level_fairness"] = experience_fairness
        
        # Anti-cheat detection accuracy
        cheat_detection_metrics = {}
        copy_paste_cases = [r for r, b in zip(results, dataset) if "copypaste" in b.id.lower()]
        if copy_paste_cases:
            detected = sum(1 for r in copy_paste_cases if r.anti_cheat_flags.get("is_copy_paste", False))
            cheat_detection_metrics["copy_paste_detection_rate"] = detected / len(copy_paste_cases)
        
        metrics["cheat_detection"] = cheat_detection_metrics
        
        # Quality level discrimination
        quality_means = {}
        for quality in AnswerQuality:
            quality_results = [r for r, b in zip(results, dataset) if b.quality_level == quality]
            if quality_results:
                quality_means[quality.value] = sum(r.actual_score for r in quality_results) / len(quality_results)
        
        # Check if scores properly discriminate between quality levels
        quality_order_correct = True
        quality_values = list(quality_means.values())
        if len(quality_values) > 1:
            # Scores should generally increase from bad to excellent
            for i in range(len(quality_values) - 1):
                if quality_values[i] > quality_values[i + 1]:
                    quality_order_correct = False
                    break
        
        metrics["quality_discrimination"] = {
            "mean_scores_by_quality": quality_means,
            "proper_ordering": quality_order_correct
        }
        
        return metrics
    
    def _save_report(self, report: BenchmarkReport):
        """Save benchmark report to file"""
        
        report_file = self.results_dir / f"{report.dataset_id}_report.json"
        
        # Convert to JSON-serializable format
        report_data = {
            "dataset_id": report.dataset_id,
            "total_cases": report.total_cases,
            "quality_distribution": report.quality_distribution,
            "overall_accuracy": report.overall_accuracy,
            "range_accuracy_by_quality": report.range_accuracy_by_quality,
            "avg_evaluation_time": report.avg_evaluation_time,
            "score_distribution": report.score_distribution,
            "fairness_metrics": report.fairness_metrics,
            "generated_at": report.generated_at.isoformat(),
            "detailed_results": [
                {
                    "benchmark_id": r.benchmark_id,
                    "quality_level": r.quality_level.value,
                    "expected_range": r.expected_range,
                    "actual_score": r.actual_score,
                    "within_range": r.within_range,
                    "score_accuracy": r.score_accuracy,
                    "evaluation_time_ms": r.evaluation_time_ms,
                    "detailed_scores": r.detailed_scores,
                    "anti_cheat_flags": r.anti_cheat_flags
                }
                for r in report.detailed_results
            ]
        }
        
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Benchmark report saved: {report_file}")
    
    def generate_summary_report(self, report: BenchmarkReport) -> str:
        """Generate human-readable summary report"""
        
        lines = [
            "📊 BENCHMARK EVALUATION REPORT",
            "=" * 50,
            f"Dataset ID: {report.dataset_id}",
            f"Total Test Cases: {report.total_cases}",
            f"Overall Accuracy: {report.overall_accuracy:.1%}",
            f"Average Evaluation Time: {report.avg_evaluation_time:.1f}ms",
            "",
            "Quality Distribution:",
        ]
        
        for quality, count in report.quality_distribution.items():
            lines.append(f"  {quality.title()}: {count} cases")
        
        lines.extend([
            "",
            "Range Accuracy by Quality:"
        ])
        
        for quality, accuracy in report.range_accuracy_by_quality.items():
            lines.append(f"  {quality.title()}: {accuracy:.1%}")
        
        lines.extend([
            "",
            "Score Statistics by Quality:"
        ])
        
        for quality, scores in report.score_distribution.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                min_score = min(scores)
                max_score = max(scores)
                lines.append(f"  {quality.title()}: avg={avg_score:.1f}, range=[{min_score:.1f}-{max_score:.1f}]")
        
        # Fairness metrics
        lines.extend([
            "",
            "Fairness Analysis:"
        ])
        
        fairness = report.fairness_metrics
        
        if "experience_level_fairness" in fairness:
            lines.append("  Experience Level Consistency:")
            for level, metrics in fairness["experience_level_fairness"].items():
                cv = metrics["coefficient_variation"]
                lines.append(f"    {level.title()}: CV={cv:.3f} (lower is more consistent)")
        
        if "quality_discrimination" in fairness:
            quality_order = fairness["quality_discrimination"]["proper_ordering"]
            status = "✅" if quality_order else "❌"
            lines.append(f"  Quality Level Discrimination: {status}")
        
        if "cheat_detection" in fairness and fairness["cheat_detection"]:
            cheat_metrics = fairness["cheat_detection"]
            if "copy_paste_detection_rate" in cheat_metrics:
                detection_rate = cheat_metrics["copy_paste_detection_rate"]
                lines.append(f"  Copy-Paste Detection Rate: {detection_rate:.1%}")
        
        return "\n".join(lines)


# Main function to create and evaluate benchmark dataset
def create_and_evaluate_benchmark():
    """Create comprehensive benchmark dataset and evaluate system performance"""
    
    print("🚀 Creating comprehensive benchmark dataset...")
    creator = BenchmarkDatasetCreator()
    dataset = creator.create_comprehensive_dataset()
    
    # Save dataset
    dataset_file = creator.save_dataset(dataset, "comprehensive_benchmark")
    print(f"📁 Dataset saved to: {dataset_file}")
    
    print(f"\n📊 Evaluating LLM performance on {len(dataset)} benchmark answers...")
    evaluator = BenchmarkEvaluator()
    report = evaluator.evaluate_dataset(dataset)
    
    # Print summary
    summary = evaluator.generate_summary_report(report)
    print(f"\n{summary}")
    
    return dataset, report


if __name__ == "__main__":
    create_and_evaluate_benchmark()