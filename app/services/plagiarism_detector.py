"""
Advanced Plagiarism Similarity Check for LLM Interview Evaluation System

This module implements intelligent plagiarism detection by comparing candidate answers
with model-generated ideal answers using sophisticated similarity metrics and NLP techniques.
Detects potential cheating through textual similarity analysis.
"""

import re
import time
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path

# For similarity calculations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy

# For additional metrics
from difflib import SequenceMatcher
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Pydantic for data models
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class SimilarityMethod(str, Enum):
    """Available similarity calculation methods"""
    COSINE_TFIDF = "cosine_tfidf"
    COSINE_SENTENCE_BERT = "cosine_sentence_bert"
    JACCARD_SIMILARITY = "jaccard_similarity"
    SEQUENCE_MATCHER = "sequence_matcher"
    SEMANTIC_SIMILARITY = "semantic_similarity"


class PlagiarismRiskLevel(str, Enum):
    """Plagiarism risk assessment levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SimilarityMetrics:
    """Container for various similarity measurements"""
    cosine_tfidf: float
    cosine_sentence_bert: float
    jaccard_similarity: float
    sequence_similarity: float
    semantic_similarity: float
    overall_similarity: float


class PlagiarismConfig(BaseModel):
    """Configuration for plagiarism detection parameters"""
    
    # Similarity thresholds
    critical_threshold: float = Field(default=0.90, ge=0.0, le=1.0, description="Critical similarity threshold")
    high_threshold: float = Field(default=0.80, ge=0.0, le=1.0, description="High similarity threshold")
    medium_threshold: float = Field(default=0.65, ge=0.0, le=1.0, description="Medium similarity threshold")
    low_threshold: float = Field(default=0.50, ge=0.0, le=1.0, description="Low similarity threshold")
    
    # Weights for different similarity methods
    tfidf_weight: float = Field(default=0.30, ge=0.0, le=1.0, description="TF-IDF cosine similarity weight")
    sentence_bert_weight: float = Field(default=0.35, ge=0.0, le=1.0, description="Sentence-BERT similarity weight")
    jaccard_weight: float = Field(default=0.15, ge=0.0, le=1.0, description="Jaccard similarity weight")
    sequence_weight: float = Field(default=0.10, ge=0.0, le=1.0, description="Sequence similarity weight")
    semantic_weight: float = Field(default=0.10, ge=0.0, le=1.0, description="Semantic similarity weight")
    
    # Text preprocessing options
    remove_stopwords: bool = Field(default=True, description="Remove stopwords before comparison")
    use_stemming: bool = Field(default=True, description="Apply stemming to words")
    use_lemmatization: bool = Field(default=True, description="Apply lemmatization to words")
    min_answer_length: int = Field(default=20, ge=1, description="Minimum answer length for meaningful comparison")
    
    # Advanced detection features
    enable_phrase_detection: bool = Field(default=True, description="Detect copied phrases")
    enable_structure_analysis: bool = Field(default=True, description="Analyze answer structure similarity")
    enable_keyword_density: bool = Field(default=True, description="Check keyword density patterns")
    
    @validator('tfidf_weight', 'sentence_bert_weight', 'jaccard_weight', 'sequence_weight', 'semantic_weight')
    def check_weights_sum_to_one(cls, v, values):
        """Ensure all weights sum approximately to 1.0"""
        if 'semantic_weight' in values:
            total = (values.get('tfidf_weight', 0) + 
                    values.get('sentence_bert_weight', 0) + 
                    values.get('jaccard_weight', 0) + 
                    values.get('sequence_weight', 0) + v)
            if abs(total - 1.0) > 0.01:
                logger.warning(f"Similarity weights sum to {total:.3f}, not 1.0")
        return v


class PlagiarismDetectionResult(BaseModel):
    """Result of plagiarism detection analysis"""
    
    risk_level: PlagiarismRiskLevel
    overall_similarity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Detailed similarity breakdowns
    similarity_metrics: Dict[str, float]
    ideal_answer: str
    flagged_sections: List[Dict[str, Any]]
    
    # Analysis results
    explanation: str
    recommendation: str
    risk_factors: List[str]
    
    # Metadata
    processing_time_ms: float
    methods_used: List[str]
    preprocessing_applied: List[str]


class AdvancedPlagiarismDetector:
    """
    Advanced plagiarism detection system using multiple similarity metrics
    and natural language processing techniques to identify potential cheating.
    """
    
    def __init__(self, config: Optional[PlagiarismConfig] = None):
        """
        Initialize the plagiarism detector
        
        Args:
            config: Configuration for detection parameters
        """
        self.config = config or PlagiarismConfig()
        
        # Initialize NLP models and tools
        self._initialize_nlp_components()
        
        # Cache for generated ideal answers
        self.ideal_answer_cache = {}
        
        logger.info("Initialized Advanced Plagiarism Detector")
        logger.info(f"Thresholds: Critical={self.config.critical_threshold:.2f}, "
                   f"High={self.config.high_threshold:.2f}, Medium={self.config.medium_threshold:.2f}")
    
    def _initialize_nlp_components(self):
        """Initialize NLP models and preprocessing tools"""
        try:
            # Initialize sentence transformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize spaCy for advanced processing
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not found, using basic processing")
                self.nlp = None
            
            # Initialize NLTK components
            try:
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('wordnet', quiet=True)
                self.stopwords = set(stopwords.words('english'))
                self.stemmer = PorterStemmer()
                self.lemmatizer = WordNetLemmatizer()
            except Exception as e:
                logger.warning(f"NLTK initialization failed: {e}")
                self.stopwords = set()
                self.stemmer = None
                self.lemmatizer = None
            
            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english' if self.config.remove_stopwords else None,
                ngram_range=(1, 3),
                lowercase=True
            )
            
            logger.info("NLP components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP components: {e}")
            raise
    
    def generate_ideal_answer(self, question: str, expected_keywords: List[str], 
                            experience_level: str = "intermediate",
                            use_cache: bool = True) -> str:
        """
        Generate an ideal answer for comparison
        
        Args:
            question: The interview question
            expected_keywords: Expected keywords in the answer
            experience_level: Expected experience level
            use_cache: Whether to use cached ideal answers
        
        Returns:
            Generated ideal answer
        """
        # Create cache key
        cache_key = hashlib.md5(
            f"{question}_{','.join(sorted(expected_keywords))}_{experience_level}".encode()
        ).hexdigest()
        
        if use_cache and cache_key in self.ideal_answer_cache:
            return self.ideal_answer_cache[cache_key]
        
        # Generate ideal answer using a comprehensive prompt
        ideal_prompt = f"""
        As an expert interviewer, provide an ideal {experience_level}-level answer to the following question.
        The answer should naturally incorporate these key concepts: {', '.join(expected_keywords)}
        
        Question: {question}
        
        Provide a clear, comprehensive answer that demonstrates strong understanding.
        Answer should be 100-200 words and technically accurate.
        """
        
        try:
            # Use OpenAI to generate ideal answer
            from openai import OpenAI
            from app.config import OPENAI_API_KEY, USE_MOCK_MODE
            
            if not USE_MOCK_MODE and OPENAI_API_KEY:
                client = OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert technical interviewer."},
                        {"role": "user", "content": ideal_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=300
                )
                ideal_answer = response.choices[0].message.content.strip()
            else:
                # Mock ideal answer for testing
                ideal_answer = self._generate_mock_ideal_answer(question, expected_keywords, experience_level)
            
            # Cache the result
            if use_cache:
                self.ideal_answer_cache[cache_key] = ideal_answer
            
            logger.info(f"Generated ideal answer for question (cached: {use_cache})")
            return ideal_answer
            
        except Exception as e:
            logger.error(f"Failed to generate ideal answer: {e}")
            return self._generate_mock_ideal_answer(question, expected_keywords, experience_level)
    
    def _generate_mock_ideal_answer(self, question: str, expected_keywords: List[str], 
                                  experience_level: str) -> str:
        """Generate a simple mock ideal answer for testing"""
        return f"""
        For a {experience_level} level candidate, this question requires understanding of {', '.join(expected_keywords[:3])}.
        The key concepts include proper implementation of these technologies and their practical applications.
        A good answer should demonstrate both theoretical knowledge and practical experience with real-world examples.
        It's important to explain the benefits and trade-offs when discussing these topics.
        """
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for similarity comparison
        
        Args:
            text: Input text to preprocess
        
        Returns:
            Preprocessed text
        """
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = text.lower().strip()
        
        # Tokenize
        tokens = word_tokenize(text) if word_tokenize else text.split()
        
        # Remove stopwords
        if self.config.remove_stopwords and self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        # Apply stemming
        if self.config.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Apply lemmatization
        if self.config.use_lemmatization and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def calculate_cosine_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity using TF-IDF vectors"""
        try:
            # Preprocess texts
            processed_text1 = self.preprocess_text(text1)
            processed_text2 = self.preprocess_text(text2)
            
            # Fit and transform texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([processed_text1, processed_text2])
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return float(similarity_matrix[0][1])
        
        except Exception as e:
            logger.warning(f"TF-IDF similarity calculation failed: {e}")
            return 0.0
    
    def calculate_sentence_bert_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity using Sentence-BERT embeddings"""
        try:
            # Generate embeddings
            embeddings = self.sentence_model.encode([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
            return float(similarity[0][0])
        
        except Exception as e:
            logger.warning(f"Sentence-BERT similarity calculation failed: {e}")
            return 0.0
    
    def calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between word sets"""
        try:
            # Preprocess and tokenize
            tokens1 = set(self.preprocess_text(text1).split())
            tokens2 = set(self.preprocess_text(text2).split())
            
            # Calculate Jaccard similarity
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            
            return intersection / union if union > 0 else 0.0
        
        except Exception as e:
            logger.warning(f"Jaccard similarity calculation failed: {e}")
            return 0.0
    
    def calculate_sequence_similarity(self, text1: str, text2: str) -> float:
        """Calculate sequence similarity using SequenceMatcher"""
        try:
            matcher = SequenceMatcher(None, text1.lower(), text2.lower())
            return matcher.ratio()
        
        except Exception as e:
            logger.warning(f"Sequence similarity calculation failed: {e}")
            return 0.0
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using spaCy if available"""
        try:
            if not self.nlp:
                return 0.0
            
            doc1 = self.nlp(text1)
            doc2 = self.nlp(text2)
            
            # Calculate similarity using spaCy's built-in similarity
            return doc1.similarity(doc2)
        
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def calculate_all_similarities(self, candidate_answer: str, ideal_answer: str) -> SimilarityMetrics:
        """
        Calculate all similarity metrics between candidate and ideal answers
        
        Args:
            candidate_answer: The candidate's answer
            ideal_answer: The ideal answer for comparison
        
        Returns:
            SimilarityMetrics with all calculated similarities
        """
        # Calculate individual similarities
        cosine_tfidf = self.calculate_cosine_tfidf_similarity(candidate_answer, ideal_answer)
        cosine_sentence_bert = self.calculate_sentence_bert_similarity(candidate_answer, ideal_answer)
        jaccard_similarity = self.calculate_jaccard_similarity(candidate_answer, ideal_answer)
        sequence_similarity = self.calculate_sequence_similarity(candidate_answer, ideal_answer)
        semantic_similarity = self.calculate_semantic_similarity(candidate_answer, ideal_answer)
        
        # Calculate weighted overall similarity
        overall_similarity = (
            cosine_tfidf * self.config.tfidf_weight +
            cosine_sentence_bert * self.config.sentence_bert_weight +
            jaccard_similarity * self.config.jaccard_weight +
            sequence_similarity * self.config.sequence_weight +
            semantic_similarity * self.config.semantic_weight
        )
        
        return SimilarityMetrics(
            cosine_tfidf=cosine_tfidf,
            cosine_sentence_bert=cosine_sentence_bert,
            jaccard_similarity=jaccard_similarity,
            sequence_similarity=sequence_similarity,
            semantic_similarity=semantic_similarity,
            overall_similarity=overall_similarity
        )
    
    def detect_flagged_sections(self, candidate_answer: str, ideal_answer: str) -> List[Dict[str, Any]]:
        """
        Detect specific sections that might be plagiarized
        
        Args:
            candidate_answer: The candidate's answer
            ideal_answer: The ideal answer for comparison
        
        Returns:
            List of flagged sections with details
        """
        flagged_sections = []
        
        # Split into sentences for analysis
        candidate_sentences = sent_tokenize(candidate_answer)
        ideal_sentences = sent_tokenize(ideal_answer)
        
        for i, candidate_sent in enumerate(candidate_sentences):
            for j, ideal_sent in enumerate(ideal_sentences):
                # Calculate similarity for each sentence pair
                sent_similarity = self.calculate_sentence_bert_similarity(candidate_sent, ideal_sent)
                
                if sent_similarity > 0.85:  # High similarity threshold for sentences
                    flagged_sections.append({
                        "candidate_sentence": candidate_sent,
                        "ideal_sentence": ideal_sent,
                        "similarity": sent_similarity,
                        "candidate_position": i,
                        "ideal_position": j,
                        "concern_level": "high" if sent_similarity > 0.95 else "medium"
                    })
        
        return flagged_sections
    
    def determine_risk_level(self, similarities: SimilarityMetrics, 
                           flagged_sections: List[Dict[str, Any]]) -> Tuple[PlagiarismRiskLevel, float, str]:
        """
        Determine plagiarism risk level based on similarity metrics
        
        Args:
            similarities: Calculated similarity metrics
            flagged_sections: List of flagged sections
        
        Returns:
            Tuple of (risk_level, confidence, explanation)
        """
        overall_sim = similarities.overall_similarity
        
        # Determine base risk level
        if overall_sim >= self.config.critical_threshold:
            risk_level = PlagiarismRiskLevel.CRITICAL
            base_confidence = 0.95
            explanation = f"Extremely high similarity ({overall_sim:.1%}) strongly suggests plagiarism"
            
        elif overall_sim >= self.config.high_threshold:
            risk_level = PlagiarismRiskLevel.HIGH
            base_confidence = 0.85
            explanation = f"High similarity ({overall_sim:.1%}) indicates potential plagiarism"
            
        elif overall_sim >= self.config.medium_threshold:
            risk_level = PlagiarismRiskLevel.MEDIUM
            base_confidence = 0.70
            explanation = f"Moderate similarity ({overall_sim:.1%}) warrants investigation"
            
        elif overall_sim >= self.config.low_threshold:
            risk_level = PlagiarismRiskLevel.LOW
            base_confidence = 0.60
            explanation = f"Some similarity ({overall_sim:.1%}) detected but likely acceptable"
            
        else:
            risk_level = PlagiarismRiskLevel.NONE
            base_confidence = 0.95
            explanation = f"Low similarity ({overall_sim:.1%}) indicates original content"
        
        # Adjust confidence based on flagged sections
        high_concern_sections = len([s for s in flagged_sections if s.get("concern_level") == "high"])
        if high_concern_sections > 0:
            base_confidence = min(base_confidence + 0.1, 0.98)
            explanation += f" with {high_concern_sections} highly similar sections"
        
        return risk_level, base_confidence, explanation
    
    def analyze_plagiarism(self, candidate_answer: str, question: str, 
                         expected_keywords: List[str], 
                         experience_level: str = "intermediate") -> PlagiarismDetectionResult:
        """
        Perform comprehensive plagiarism analysis
        
        Args:
            candidate_answer: The candidate's answer to analyze
            question: The original question
            expected_keywords: Expected keywords in the answer
            experience_level: Expected experience level
        
        Returns:
            Comprehensive plagiarism detection result
        """
        start_time = time.time()
        
        try:
            # Check minimum answer length
            if len(candidate_answer.split()) < self.config.min_answer_length:
                return PlagiarismDetectionResult(
                    risk_level=PlagiarismRiskLevel.NONE,
                    overall_similarity=0.0,
                    confidence=0.9,
                    similarity_metrics={},
                    ideal_answer="",
                    flagged_sections=[],
                    explanation="Answer too short for meaningful plagiarism analysis",
                    recommendation="No action needed",
                    risk_factors=[],
                    processing_time_ms=(time.time() - start_time) * 1000,
                    methods_used=[],
                    preprocessing_applied=[]
                )
            
            # Generate ideal answer
            ideal_answer = self.generate_ideal_answer(question, expected_keywords, experience_level)
            
            # Calculate all similarity metrics
            similarities = self.calculate_all_similarities(candidate_answer, ideal_answer)
            
            # Detect flagged sections
            flagged_sections = self.detect_flagged_sections(candidate_answer, ideal_answer)
            
            # Determine risk level
            risk_level, confidence, explanation = self.determine_risk_level(similarities, flagged_sections)
            
            # Generate recommendation
            recommendations = {
                PlagiarismRiskLevel.CRITICAL: "Immediate review required - likely plagiarism detected",
                PlagiarismRiskLevel.HIGH: "Manual review recommended - high plagiarism risk",
                PlagiarismRiskLevel.MEDIUM: "Consider additional evaluation - moderate risk",
                PlagiarismRiskLevel.LOW: "Monitor for patterns - low risk",
                PlagiarismRiskLevel.NONE: "No action needed - original content"
            }
            
            # Identify risk factors
            risk_factors = []
            if similarities.overall_similarity > 0.8:
                risk_factors.append("Very high overall similarity")
            if similarities.sentence_bert_similarity > 0.85:
                risk_factors.append("High semantic similarity")
            if len(flagged_sections) > 2:
                risk_factors.append("Multiple similar sections detected")
            if similarities.sequence_similarity > 0.9:
                risk_factors.append("High sequential similarity")
            
            processing_time = (time.time() - start_time) * 1000
            
            return PlagiarismDetectionResult(
                risk_level=risk_level,
                overall_similarity=similarities.overall_similarity,
                confidence=confidence,
                similarity_metrics={
                    "cosine_tfidf": similarities.cosine_tfidf,
                    "cosine_sentence_bert": similarities.cosine_sentence_bert,
                    "jaccard_similarity": similarities.jaccard_similarity,
                    "sequence_similarity": similarities.sequence_similarity,
                    "semantic_similarity": similarities.semantic_similarity
                },
                ideal_answer=ideal_answer,
                flagged_sections=flagged_sections,
                explanation=explanation,
                recommendation=recommendations[risk_level],
                risk_factors=risk_factors,
                processing_time_ms=processing_time,
                methods_used=["tfidf", "sentence_bert", "jaccard", "sequence", "semantic"],
                preprocessing_applied=["lowercase", "tokenization"] + 
                                   (["stopword_removal"] if self.config.remove_stopwords else []) +
                                   (["stemming"] if self.config.use_stemming else []) +
                                   (["lemmatization"] if self.config.use_lemmatization else [])
            )
            
        except Exception as e:
            logger.error(f"Plagiarism analysis failed: {e}")
            
            # Return safe fallback result
            return PlagiarismDetectionResult(
                risk_level=PlagiarismRiskLevel.LOW,
                overall_similarity=0.0,
                confidence=0.1,
                similarity_metrics={},
                ideal_answer="",
                flagged_sections=[],
                explanation=f"Analysis failed with error: {str(e)}",
                recommendation="Manual review recommended due to analysis failure",
                risk_factors=["analysis_failure"],
                processing_time_ms=(time.time() - start_time) * 1000,
                methods_used=[],
                preprocessing_applied=[]
            )


# Example usage and testing functions
def example_usage():
    """Demonstrate plagiarism detection functionality"""
    
    print("🔍 PLAGIARISM DETECTION EXAMPLE")
    print("=" * 50)
    
    # Initialize detector
    detector = AdvancedPlagiarismDetector()
    
    # Sample question and answers
    question = "Explain the difference between REST and GraphQL APIs."
    expected_keywords = ["REST", "HTTP", "GraphQL", "query", "endpoint"]
    
    # Test cases
    test_cases = [
        {
            "name": "Original Answer",
            "answer": "REST uses multiple endpoints with HTTP methods like GET, POST. GraphQL uses single endpoint with custom query language for flexible data fetching.",
            "expected_risk": "NONE"
        },
        {
            "name": "Highly Similar Answer", 
            "answer": "REST APIs utilize HTTP methods such as GET, POST, PUT, DELETE for different operations on resources. GraphQL employs a query language that allows clients to request specific data through a single endpoint.",
            "expected_risk": "HIGH"
        },
        {
            "name": "Partially Similar Answer",
            "answer": "REST follows resource-based architecture with standard HTTP methods. GraphQL provides more flexibility by allowing clients to specify exactly what data they need in a single request.",
            "expected_risk": "MEDIUM"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['name']}")
        print("-" * 30)
        
        result = detector.analyze_plagiarism(
            candidate_answer=test_case["answer"],
            question=question,
            expected_keywords=expected_keywords
        )
        
        print(f"📊 Risk Level: {result.risk_level}")
        print(f"📈 Overall Similarity: {result.overall_similarity:.1%}")
        print(f"🎯 Confidence: {result.confidence:.1%}")
        print(f"💡 Explanation: {result.explanation}")
        print(f"📋 Recommendation: {result.recommendation}")
        
        if result.flagged_sections:
            print(f"🚨 Flagged Sections: {len(result.flagged_sections)}")
            for section in result.flagged_sections[:2]:  # Show first 2
                print(f"   Similarity: {section['similarity']:.1%}")
                print(f"   Candidate: {section['candidate_sentence'][:60]}...")
        
        print(f"⏱️  Processing Time: {result.processing_time_ms:.1f}ms")


if __name__ == "__main__":
    example_usage()