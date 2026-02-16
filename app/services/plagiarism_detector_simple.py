"""
Simplified Plagiarism Detector for testing purposes
This bypasses the sklearn dependency issue temporarily
"""

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class PlagiarismAnalysis(BaseModel):
    """Simplified plagiarism analysis result"""
    similarity_score: float = 0.0
    is_potential_plagiarism: bool = False
    confidence_level: str = "unknown"
    matched_patterns: List[str] = []
    risk_indicators: List[str] = []
    
class AdvancedPlagiarismDetector:
    """Simplified version for testing - bypasses sklearn issues"""
    
    def __init__(self):
        logger.info("Initialized AdvancedPlagiarismDetector (simplified version)")
    
    def analyze_plagiarism(self, answer: str, question: str = "", context: Dict = None) -> PlagiarismAnalysis:
        """Simple plagiarism analysis without ML dependencies"""
        # Basic word overlap check
        if not answer or len(answer.strip()) < 10:
            return PlagiarismAnalysis(
                similarity_score=0.0,
                is_potential_plagiarism=False,
                confidence_level="low",
                risk_indicators=["Answer too short for analysis"]
            )
        
        # Simple heuristics for now
        common_phrases = [
            "according to my knowledge", 
            "in my opinion", 
            "based on my understanding",
            "copy and paste",
            "ctrl+c ctrl+v"
        ]
        
        answer_lower = answer.lower()
        matched_patterns = [phrase for phrase in common_phrases if phrase in answer_lower]
        
        # Very basic similarity scoring
        similarity_score = min(len(matched_patterns) * 0.2, 0.9)
        is_plagiarism = similarity_score > 0.5
        
        return PlagiarismAnalysis(
            similarity_score=similarity_score,
            is_potential_plagiarism=is_plagiarism,
            confidence_level="medium" if is_plagiarism else "low",
            matched_patterns=matched_patterns,
            risk_indicators=["Pattern-based analysis only"] if is_plagiarism else []
        )
    
    def get_detector_info(self) -> Dict:
        """Return detector information"""
        return {
            "version": "1.0-simplified",
            "methods": ["pattern_matching"],
            "status": "active"
        }