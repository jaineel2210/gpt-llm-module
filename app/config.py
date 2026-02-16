import os
from dotenv import load_dotenv
from typing import Dict, List

load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"
USE_MOCK_MODE = os.getenv("USE_MOCK_MODE", "false").lower() == "true"

# LLM Settings
LLM_CONFIG = {
    "model": MODEL_NAME,
    "temperature": 0.2,  # Low temperature for consistent evaluation
    "max_tokens": 1500,  # Increased for comprehensive responses
    "top_p": 0.9,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1
}

# Scoring Configuration
DEFAULT_WEIGHTS = {
    "technical_accuracy": 0.40,
    "concept_clarity": 0.25,
    "keyword_coverage": 0.20,
    "communication": 0.15
}

# Anti-Cheating Detection Thresholds
ANTI_CHEAT_CONFIG = {
    "min_answer_length": 10,  # Minimum words for valid answer
    "max_technical_score_for_short": 6,  # Max score if answer too short
    "copy_paste_keywords": [
        "according to textbook", "as mentioned in", "reference:", 
        "bibliography", "cited from", "source:", "wikipedia"
    ],
    "ai_generated_indicators": [
        "artificial intelligence", "as an ai", "i don't have personal",
        "i cannot", "i'm designed to", "my programming"
    ],
    "robotic_indicators": {
        "min_filler_words": 2,  # Expect some natural speech
        "max_formal_ratio": 0.9,  # Too formal indicates copy-paste
        "required_contractions": True  # Natural speech has contractions
    }
}

# Risk Assessment Levels
RISK_LEVELS = {
    "low": {"min_score": 0.0, "max_cheat_prob": 0.2},
    "medium": {"min_score": 0.2, "max_cheat_prob": 0.6},
    "high": {"min_score": 0.6, "max_cheat_prob": 1.0}
}

# Error Handling
ERROR_CONFIG = {
    "max_retries": 3,
    "timeout_seconds": 30,
    "fallback_to_mock": True,
    "log_failures": True
}
