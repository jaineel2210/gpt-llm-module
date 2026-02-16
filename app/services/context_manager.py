"""
Multi-Turn Interview Context Manager for LLM Interview Evaluation System

This module implements intelligent multi-turn conversation context management
that enables the LLM to remember previous answers and provide context-aware
evaluation considering the full interview flow.
"""

import json
import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import pickle

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ContextStorageType(str, Enum):
    """Available context storage types"""
    MEMORY = "memory"
    FILE = "file"
    DATABASE = "database"  # For future implementation


class ConsistencyLevel(str, Enum):
    """Consistency assessment levels"""
    HIGHLY_CONSISTENT = "highly_consistent"
    CONSISTENT = "consistent"
    SOMEWHAT_CONSISTENT = "somewhat_consistent"
    INCONSISTENT = "inconsistent"
    HIGHLY_INCONSISTENT = "highly_inconsistent"


@dataclass
class QuestionAnswerPair:
    """Individual question-answer pair with metadata"""
    question: str
    answer: str
    question_type: str
    score: float
    timestamp: datetime
    keywords_covered: List[str]
    evaluation_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuestionAnswerPair':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class ContextConfig(BaseModel):
    """Configuration for multi-turn context management"""
    
    # Context window settings
    max_context_pairs: int = Field(default=10, ge=1, le=50, description="Maximum Q&A pairs to maintain in context")
    context_window_minutes: int = Field(default=60, ge=5, le=480, description="Time window for context relevance")
    
    # Storage settings
    storage_type: ContextStorageType = Field(default=ContextStorageType.MEMORY, description="Context storage method")
    storage_path: Optional[str] = Field(default=None, description="Path for file-based storage")
    auto_save: bool = Field(default=True, description="Automatically save context changes")
    
    # Context analysis settings
    enable_consistency_analysis: bool = Field(default=True, description="Analyze answer consistency across turns")
    consistency_weight: float = Field(default=0.15, ge=0.0, le=1.0, description="Weight for consistency in evaluation")
    enable_topic_tracking: bool = Field(default=True, description="Track topic progression across questions")
    
    # Performance settings
    enable_context_compression: bool = Field(default=True, description="Compress old context for efficiency")
    compression_threshold: int = Field(default=5, ge=2, le=20, description="Number of pairs before compression")


class ConsistencyAnalysis(BaseModel):
    """Analysis of answer consistency across interview turns"""
    
    overall_consistency: ConsistencyLevel
    consistency_score: float = Field(ge=0.0, le=1.0)
    
    # Detailed analysis
    technical_consistency: float = Field(ge=0.0, le=1.0, description="Technical knowledge consistency")
    conceptual_consistency: float = Field(ge=0.0, le=1.0, description="Conceptual understanding consistency")
    style_consistency: float = Field(ge=0.0, le=1.0, description="Communication style consistency")
    
    # Findings
    inconsistencies_found: List[Dict[str, Any]]
    supporting_evidence: List[Dict[str, Any]]
    
    explanation: str
    recommendation: str


class ContextState(BaseModel):
    """Current state of the interview context"""
    
    interview_id: str
    candidate_id: Optional[str]
    
    # Context data
    question_history: List[Dict[str, Any]]
    topic_progression: List[str]
    covered_concepts: List[str]
    
    # Analysis results
    consistency_analysis: Optional[ConsistencyAnalysis]
    context_summary: str
    
    # Metadata
    created_at: datetime
    updated_at: datetime
    total_questions: int
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class MultiTurnContextManager:
    """
    Advanced multi-turn interview context manager that maintains conversation
    history and provides context-aware evaluation capabilities.
    """
    
    def __init__(self, config: Optional[ContextConfig] = None):
        """
        Initialize the context manager
        
        Args:
            config: Configuration for context management
        """
        self.config = config or ContextConfig()
        
        # Context storage
        self.active_contexts: Dict[str, ContextState] = {}
        
        # Initialize storage backend
        self._initialize_storage()
        
        logger.info("Initialized Multi-Turn Context Manager")
        logger.info(f"Max context pairs: {self.config.max_context_pairs}, "
                   f"Window: {self.config.context_window_minutes} minutes")
    
    def _initialize_storage(self):
        """Initialize the chosen storage backend"""
        if self.config.storage_type == ContextStorageType.FILE:
            if self.config.storage_path:
                self.storage_path = Path(self.config.storage_path)
                self.storage_path.mkdir(parents=True, exist_ok=True)
            else:
                self.storage_path = Path("./context_storage")
                self.storage_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"File storage initialized at: {self.storage_path}")
        
        elif self.config.storage_type == ContextStorageType.MEMORY:
            # Memory storage is ready by default
            logger.info("Memory storage initialized")
        
        elif self.config.storage_type == ContextStorageType.DATABASE:
            # Future implementation
            logger.warning("Database storage not yet implemented, falling back to memory")
            self.config.storage_type = ContextStorageType.MEMORY
    
    def create_context(self, interview_id: str, candidate_id: Optional[str] = None) -> ContextState:
        """
        Create a new interview context
        
        Args:
            interview_id: Unique interview identifier
            candidate_id: Optional candidate identifier
        
        Returns:
            New ContextState
        """
        context = ContextState(
            interview_id=interview_id,
            candidate_id=candidate_id,
            question_history=[],
            topic_progression=[],
            covered_concepts=[],
            consistency_analysis=None,
            context_summary="Interview context initialized",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            total_questions=0
        )
        
        self.active_contexts[interview_id] = context
        
        if self.config.auto_save:
            self._save_context(context)
        
        logger.info(f"Created new context for interview: {interview_id}")
        return context
    
    def add_question_answer(self, 
                          interview_id: str,
                          question: str,
                          answer: str,
                          question_type: str,
                          score: float,
                          keywords_covered: List[str],
                          evaluation_metadata: Optional[Dict[str, Any]] = None) -> ContextState:
        """
        Add a new question-answer pair to the context
        
        Args:
            interview_id: Interview identifier
            question: The question asked
            answer: The candidate's answer
            question_type: Type of question (technical, behavioral, etc.)
            score: Evaluation score for the answer
            keywords_covered: Keywords found in the answer
            evaluation_metadata: Additional evaluation metadata
        
        Returns:
            Updated ContextState
        """
        # Get or create context
        if interview_id not in self.active_contexts:
            context = self.create_context(interview_id)
        else:
            context = self.active_contexts[interview_id]
        
        # Create Q&A pair
        qa_pair = QuestionAnswerPair(
            question=question,
            answer=answer,
            question_type=question_type,
            score=score,
            timestamp=datetime.utcnow(),
            keywords_covered=keywords_covered,
            evaluation_metadata=evaluation_metadata or {}
        )
        
        # Add to history
        context.question_history.append(qa_pair.to_dict())
        context.total_questions += 1
        context.updated_at = datetime.utcnow()
        
        # Update topic progression
        self._update_topic_progression(context, question_type, keywords_covered)
        
        # Update covered concepts
        context.covered_concepts.extend(keywords_covered)
        context.covered_concepts = list(set(context.covered_concepts))  # Remove duplicates
        
        # Maintain context window
        context = self._maintain_context_window(context)
        
        # Perform consistency analysis if enabled
        if self.config.enable_consistency_analysis and len(context.question_history) > 1:
            context.consistency_analysis = self._analyze_consistency(context)
        
        # Update context summary
        context.context_summary = self._generate_context_summary(context)
        
        # Save if auto-save enabled
        if self.config.auto_save:
            self._save_context(context)
        
        self.active_contexts[interview_id] = context
        
        logger.info(f"Added Q&A pair to context {interview_id}. Total questions: {context.total_questions}")
        return context
    
    def get_context_for_evaluation(self, interview_id: str) -> Optional[str]:
        """
        Get formatted context for LLM evaluation
        
        Args:
            interview_id: Interview identifier
        
        Returns:
            Formatted context string for LLM prompt
        """
        if interview_id not in self.active_contexts:
            return None
        
        context = self.active_contexts[interview_id]
        
        # Build context string
        context_parts = []
        
        # Interview summary
        context_parts.append(f"INTERVIEW CONTEXT (Total Questions: {context.total_questions})")
        context_parts.append(f"Topics Covered: {', '.join(context.topic_progression[-5:])}")  # Last 5 topics
        context_parts.append(f"Key Concepts: {', '.join(context.covered_concepts[-10:])}")  # Last 10 concepts
        
        # Recent question history (last 3-5 pairs)
        recent_history = context.question_history[-3:]
        if recent_history:
            context_parts.append("\nRECENT CONVERSATION HISTORY:")
            for i, qa_dict in enumerate(recent_history, 1):
                qa = QuestionAnswerPair.from_dict(qa_dict)
                context_parts.append(f"\nQ{i}: {qa.question}")
                context_parts.append(f"A{i}: {qa.answer[:200]}..." if len(qa.answer) > 200 else f"A{i}: {qa.answer}")
                context_parts.append(f"Score: {qa.score:.1f}, Type: {qa.question_type}")
        
        # Consistency analysis if available
        if context.consistency_analysis:
            context_parts.append(f"\nCONSISTENCY ANALYSIS:")
            context_parts.append(f"Overall Consistency: {context.consistency_analysis.overall_consistency}")
            context_parts.append(f"Consistency Score: {context.consistency_analysis.consistency_score:.2f}")
            if context.consistency_analysis.inconsistencies_found:
                context_parts.append(f"Inconsistencies Detected: {len(context.consistency_analysis.inconsistencies_found)}")
        
        context_parts.append(f"\nCONTEXT SUMMARY: {context.context_summary}")
        
        return "\n".join(context_parts)
    
    def _update_topic_progression(self, context: ContextState, question_type: str, keywords: List[str]):
        """Update topic progression tracking"""
        if question_type not in context.topic_progression:
            context.topic_progression.append(question_type)
        
        # Add significant keywords as topics
        significant_keywords = [kw for kw in keywords if len(kw) > 3]  # Filter short words
        for keyword in significant_keywords[:3]:  # Add up to 3 keywords
            if keyword.lower() not in [t.lower() for t in context.topic_progression]:
                context.topic_progression.append(keyword)
    
    def _maintain_context_window(self, context: ContextState) -> ContextState:
        """Maintain context within configured limits"""
        # Remove old entries beyond time window
        cutoff_time = datetime.utcnow() - timedelta(minutes=self.config.context_window_minutes)
        context.question_history = [
            qa_dict for qa_dict in context.question_history
            if datetime.fromisoformat(qa_dict['timestamp']) > cutoff_time
        ]
        
        # Limit total entries
        if len(context.question_history) > self.config.max_context_pairs:
            # Keep most recent entries
            context.question_history = context.question_history[-self.config.max_context_pairs:]
        
        return context
    
    def _analyze_consistency(self, context: ContextState) -> ConsistencyAnalysis:
        """
        Analyze consistency across answers in the interview
        
        Args:
            context: Current context state
        
        Returns:
            ConsistencyAnalysis results
        """
        if len(context.question_history) < 2:
            return ConsistencyAnalysis(
                overall_consistency=ConsistencyLevel.CONSISTENT,
                consistency_score=1.0,
                technical_consistency=1.0,
                conceptual_consistency=1.0,
                style_consistency=1.0,
                inconsistencies_found=[],
                supporting_evidence=[],
                explanation="Insufficient data for consistency analysis",
                recommendation="Continue interview to assess consistency"
            )
        
        try:
            # Analyze different aspects of consistency
            technical_consistency = self._analyze_technical_consistency(context)
            conceptual_consistency = self._analyze_conceptual_consistency(context)
            style_consistency = self._analyze_style_consistency(context)
            
            # Calculate overall consistency score
            overall_score = (
                technical_consistency * 0.4 +
                conceptual_consistency * 0.4 +
                style_consistency * 0.2
            )
            
            # Determine consistency level
            if overall_score >= 0.85:
                level = ConsistencyLevel.HIGHLY_CONSISTENT
            elif overall_score >= 0.70:
                level = ConsistencyLevel.CONSISTENT
            elif overall_score >= 0.55:
                level = ConsistencyLevel.SOMEWHAT_CONSISTENT
            elif overall_score >= 0.40:
                level = ConsistencyLevel.INCONSISTENT
            else:
                level = ConsistencyLevel.HIGHLY_INCONSISTENT
            
            # Identify inconsistencies and evidence
            inconsistencies, evidence = self._identify_consistency_issues(context, overall_score)
            
            # Generate explanation and recommendation
            explanation = self._generate_consistency_explanation(level, overall_score, inconsistencies)
            recommendation = self._generate_consistency_recommendation(level, inconsistencies)
            
            return ConsistencyAnalysis(
                overall_consistency=level,
                consistency_score=overall_score,
                technical_consistency=technical_consistency,
                conceptual_consistency=conceptual_consistency,
                style_consistency=style_consistency,
                inconsistencies_found=inconsistencies,
                supporting_evidence=evidence,
                explanation=explanation,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Consistency analysis failed: {e}")
            return ConsistencyAnalysis(
                overall_consistency=ConsistencyLevel.SOMEWHAT_CONSISTENT,
                consistency_score=0.6,
                technical_consistency=0.6,
                conceptual_consistency=0.6,
                style_consistency=0.6,
                inconsistencies_found=[],
                supporting_evidence=[],
                explanation=f"Analysis failed: {str(e)}",
                recommendation="Manual review recommended"
            )
    
    def _analyze_technical_consistency(self, context: ContextState) -> float:
        """Analyze consistency of technical knowledge across answers"""
        scores = []
        for qa_dict in context.question_history:
            if qa_dict['question_type'] in ['technical', 'coding']:
                scores.append(qa_dict['score'])
        
        if len(scores) < 2:
            return 1.0
        
        # Check score variance (lower variance = higher consistency)
        import statistics
        score_variance = statistics.variance(scores)
        
        # Convert variance to consistency score (0-1)
        # Lower variance = higher consistency
        consistency_score = max(0.0, 1.0 - (score_variance / 10))
        return min(1.0, consistency_score)
    
    def _analyze_conceptual_consistency(self, context: ContextState) -> float:
        """Analyze conceptual understanding consistency"""
        # Check for concept overlap and contradiction
        all_concepts = []
        for qa_dict in context.question_history:
            all_concepts.extend(qa_dict['keywords_covered'])
        
        if len(all_concepts) < 4:
            return 1.0
        
        # Simple heuristic: consistent use of related concepts
        concept_frequency = {}
        for concept in all_concepts:
            concept_frequency[concept] = concept_frequency.get(concept, 0) + 1
        
        # Higher reuse of concepts suggests consistency
        reused_concepts = [k for k, v in concept_frequency.items() if v > 1]
        consistency_ratio = len(reused_concepts) / len(set(all_concepts))
        
        return min(1.0, consistency_ratio * 2)  # Scale up to make more sensitive
    
    def _analyze_style_consistency(self, context: ContextState) -> float:
        """Analyze communication style consistency"""
        if len(context.question_history) < 2:
            return 1.0
        
        # Simple heuristics for style consistency
        answer_lengths = []
        for qa_dict in context.question_history:
            answer_lengths.append(len(qa_dict['answer'].split()))
        
        # Check length consistency as a proxy for style
        import statistics
        if len(answer_lengths) > 1:
            length_variance = statistics.variance(answer_lengths)
            avg_length = statistics.mean(answer_lengths)
            
            # Normalize variance by average length
            normalized_variance = length_variance / max(avg_length, 1)
            
            # Convert to consistency score
            consistency_score = max(0.0, 1.0 - (normalized_variance / 100))
            return min(1.0, consistency_score)
        
        return 1.0
    
    def _identify_consistency_issues(self, context: ContextState, overall_score: float) -> Tuple[List[Dict], List[Dict]]:
        """Identify specific consistency issues and supporting evidence"""
        inconsistencies = []
        evidence = []
        
        # Score variance analysis
        scores = [qa_dict['score'] for qa_dict in context.question_history]
        if len(scores) > 1:
            import statistics
            score_variance = statistics.variance(scores)
            if score_variance > 5.0:  # High variance threshold
                inconsistencies.append({
                    "type": "score_variance",
                    "description": f"High score variance ({score_variance:.2f}) across answers",
                    "severity": "medium",
                    "affected_questions": list(range(len(scores)))
                })
        
        # Topic jumping analysis
        technical_questions = [i for i, qa_dict in enumerate(context.question_history) 
                             if qa_dict['question_type'] == 'technical']
        if len(technical_questions) > 1:
            technical_scores = [context.question_history[i]['score'] for i in technical_questions]
            if max(technical_scores) - min(technical_scores) > 3.0:
                inconsistencies.append({
                    "type": "technical_inconsistency",
                    "description": "Inconsistent performance on technical questions",
                    "severity": "high",
                    "score_range": f"{min(technical_scores):.1f} - {max(technical_scores):.1f}"
                })
        
        # Supporting evidence for consistency
        if overall_score > 0.7:
            evidence.append({
                "type": "consistent_performance",
                "description": "Consistent performance across question types",
                "support_strength": "strong"
            })
        
        if len(context.covered_concepts) > 5:
            evidence.append({
                "type": "concept_coverage",
                "description": f"Good concept coverage ({len(context.covered_concepts)} concepts)",
                "support_strength": "medium"
            })
        
        return inconsistencies, evidence
    
    def _generate_consistency_explanation(self, level: ConsistencyLevel, score: float, 
                                        inconsistencies: List[Dict]) -> str:
        """Generate human-readable consistency explanation"""
        if level == ConsistencyLevel.HIGHLY_CONSISTENT:
            return f"Excellent consistency across answers (score: {score:.2f}). Candidate shows stable knowledge and communication patterns."
        elif level == ConsistencyLevel.CONSISTENT:
            return f"Good consistency overall (score: {score:.2f}). Minor variations are within normal range."
        elif level == ConsistencyLevel.SOMEWHAT_CONSISTENT:
            return f"Moderate consistency (score: {score:.2f}). Some variations detected but generally acceptable."
        elif level == ConsistencyLevel.INCONSISTENT:
            return f"Concerning inconsistencies detected (score: {score:.2f}). Performance varies significantly across answers."
        else:
            return f"Major inconsistencies found (score: {score:.2f}). Significant variations in knowledge or communication style."
    
    def _generate_consistency_recommendation(self, level: ConsistencyLevel, 
                                           inconsistencies: List[Dict]) -> str:
        """Generate consistency-based recommendations"""
        if level in [ConsistencyLevel.HIGHLY_CONSISTENT, ConsistencyLevel.CONSISTENT]:
            return "No action needed. Candidate demonstrates consistent performance."
        elif level == ConsistencyLevel.SOMEWHAT_CONSISTENT:
            return "Monitor for patterns. Consider follow-up questions in areas with variations."
        elif level == ConsistencyLevel.INCONSISTENT:
            return "Investigate inconsistencies. Ask clarifying questions about performance variations."
        else:
            return "Significant concerns. Recommend additional evaluation or targeted follow-up questions."
    
    def _generate_context_summary(self, context: ContextState) -> str:
        """Generate a summary of the current interview context"""
        if context.total_questions == 0:
            return "Interview not yet started"
        elif context.total_questions == 1:
            return f"Interview started with {context.question_history[0]['question_type']} question"
        else:
            recent_scores = [qa_dict['score'] for qa_dict in context.question_history[-3:]]
            avg_recent_score = sum(recent_scores) / len(recent_scores)
            
            topics = list(set(context.topic_progression[-5:]))
            
            summary = f"Interview in progress: {context.total_questions} questions, "
            summary += f"recent average score: {avg_recent_score:.1f}, "
            summary += f"covering {len(topics)} topics"
            
            if context.consistency_analysis:
                summary += f", consistency: {context.consistency_analysis.overall_consistency}"
            
            return summary
    
    def _save_context(self, context: ContextState):
        """Save context to configured storage backend"""
        try:
            if self.config.storage_type == ContextStorageType.FILE:
                filename = f"context_{context.interview_id}.json"
                filepath = self.storage_path / filename
                
                # Convert to dict for JSON serialization
                context_dict = context.dict()
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(context_dict, f, indent=2, default=str)
                
                logger.debug(f"Saved context to file: {filepath}")
                
            elif self.config.storage_type == ContextStorageType.MEMORY:
                # Already stored in self.active_contexts
                pass
        
        except Exception as e:
            logger.error(f"Failed to save context: {e}")
    
    def load_context(self, interview_id: str) -> Optional[ContextState]:
        """
        Load context from storage
        
        Args:
            interview_id: Interview identifier
        
        Returns:
            Loaded ContextState or None if not found
        """
        # Check active contexts first
        if interview_id in self.active_contexts:
            return self.active_contexts[interview_id]
        
        try:
            if self.config.storage_type == ContextStorageType.FILE:
                filename = f"context_{interview_id}.json"
                filepath = self.storage_path / filename
                
                if filepath.exists():
                    with open(filepath, 'r', encoding='utf-8') as f:
                        context_dict = json.load(f)
                    
                    # Parse datetime strings back to datetime objects
                    context_dict['created_at'] = datetime.fromisoformat(context_dict['created_at'])
                    context_dict['updated_at'] = datetime.fromisoformat(context_dict['updated_at'])
                    
                    context = ContextState(**context_dict)
                    self.active_contexts[interview_id] = context
                    
                    logger.info(f"Loaded context from file: {filepath}")
                    return context
        
        except Exception as e:
            logger.error(f"Failed to load context: {e}")
        
        return None
    
    def get_context_statistics(self, interview_id: str) -> Dict[str, Any]:
        """
        Get statistics about the interview context
        
        Args:
            interview_id: Interview identifier
        
        Returns:
            Dictionary with context statistics
        """
        if interview_id not in self.active_contexts:
            return {"error": "Context not found"}
        
        context = self.active_contexts[interview_id]
        
        # Calculate statistics
        scores = [qa_dict['score'] for qa_dict in context.question_history]
        question_types = [qa_dict['question_type'] for qa_dict in context.question_history]
        
        stats = {
            "interview_id": interview_id,
            "total_questions": context.total_questions,
            "duration_minutes": (context.updated_at - context.created_at).total_seconds() / 60,
            "average_score": sum(scores) / len(scores) if scores else 0,
            "score_range": {"min": min(scores), "max": max(scores)} if scores else {},
            "question_types": list(set(question_types)),
            "topics_covered": len(context.covered_concepts),
            "context_summary": context.context_summary
        }
        
        if context.consistency_analysis:
            stats["consistency"] = {
                "level": context.consistency_analysis.overall_consistency,
                "score": context.consistency_analysis.consistency_score,
                "inconsistencies": len(context.consistency_analysis.inconsistencies_found)
            }
        
        return stats


# Example usage and testing functions
def example_usage():
    """Demonstrate multi-turn context functionality"""
    
    print("🔄 MULTI-TURN INTERVIEW CONTEXT EXAMPLE")
    print("=" * 50)
    
    # Initialize context manager
    config = ContextConfig(
        max_context_pairs=5,
        context_window_minutes=30,
        enable_consistency_analysis=True
    )
    
    context_manager = MultiTurnContextManager(config)
    
    # Simulate an interview
    interview_id = "INT_001_20241218"
    
    # Create context
    context = context_manager.create_context(interview_id, "candidate_123")
    print(f"📋 Created context for interview: {interview_id}")
    
    # Simulate Q&A pairs
    qa_pairs = [
        {
            "question": "What is Object-Oriented Programming?",
            "answer": "OOP is a programming paradigm based on objects that contain data and methods. Key principles include encapsulation, inheritance, and polymorphism.",
            "question_type": "technical",
            "score": 8.5,
            "keywords": ["OOP", "objects", "encapsulation", "inheritance", "polymorphism"]
        },
        {
            "question": "Explain inheritance in OOP",
            "answer": "Inheritance allows classes to inherit properties and methods from parent classes. It enables code reuse and creates hierarchical class relationships.",
            "question_type": "technical", 
            "score": 8.0,
            "keywords": ["inheritance", "classes", "parent", "code reuse"]
        },
        {
            "question": "How do you handle conflicts in a team?",
            "answer": "I believe in open communication and finding common ground. I listen to all perspectives and work towards solutions that benefit the team.",
            "question_type": "behavioral",
            "score": 7.5,  # Different score to test consistency
            "keywords": ["communication", "team", "conflict resolution"]
        }
    ]
    
    # Add each Q&A pair
    for i, qa in enumerate(qa_pairs, 1):
        print(f"\n➕ Adding Q&A pair {i}")
        
        context = context_manager.add_question_answer(
            interview_id=interview_id,
            question=qa["question"],
            answer=qa["answer"], 
            question_type=qa["question_type"],
            score=qa["score"],
            keywords_covered=qa["keywords"]
        )
        
        print(f"   Questions so far: {context.total_questions}")
        print(f"   Topics: {', '.join(context.topic_progression)}")
        
        # Get context for evaluation
        evaluation_context = context_manager.get_context_for_evaluation(interview_id)
        if evaluation_context:
            print(f"   Context length: {len(evaluation_context)} characters")
    
    # Show final consistency analysis
    print(f"\n📊 Final Analysis:")
    if context.consistency_analysis:
        ca = context.consistency_analysis
        print(f"   Consistency Level: {ca.overall_consistency}")
        print(f"   Consistency Score: {ca.consistency_score:.2f}")
        print(f"   Technical Consistency: {ca.technical_consistency:.2f}")
        print(f"   Conceptual Consistency: {ca.conceptual_consistency:.2f}")
        print(f"   Style Consistency: {ca.style_consistency:.2f}")
        print(f"   Explanation: {ca.explanation}")
        if ca.inconsistencies_found:
            print(f"   Inconsistencies: {len(ca.inconsistencies_found)}")
            for inc in ca.inconsistencies_found:
                print(f"      - {inc['description']}")
    
    # Show context statistics
    stats = context_manager.get_context_statistics(interview_id)
    print(f"\n📈 Statistics:")
    print(f"   Average Score: {stats['average_score']:.2f}")
    print(f"   Duration: {stats['duration_minutes']:.1f} minutes")
    print(f"   Question Types: {', '.join(stats['question_types'])}")
    print(f"   Topics Covered: {stats['topics_covered']}")


if __name__ == "__main__":
    example_usage()