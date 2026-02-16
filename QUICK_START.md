# 🚀 Quick Start Guide - Advanced LLM Evaluation System

## Installation & Setup

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure Environment**:
```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
echo "USE_MOCK_MODE=false" >> .env
```

3. **Run the Application**:
```bash
uvicorn app.main:app --reload --port 8000
```

---

## 🧪 Quick Test

### Test the New Comprehensive System:
```python
import requests

# Test data
evaluation_data = {
    "question": "Explain machine learning overfitting.",
    "candidate_answer": "Overfitting occurs when a model learns training data too specifically and fails to generalize to new data.",
    "expected_keywords": ["overfitting", "generalization", "training data", "model"],
    "experience_level": "intermediate",
    "question_type": "technical",
    "context": "Data Science interview"
}

# Call comprehensive evaluation
response = requests.post(
    "http://localhost:8000/evaluate/comprehensive",
    json=evaluation_data
)

result = response.json()
print(f"Final Score: {result['scores']['final_score']}/10")
print(f"Risk Flag: {result['anti_cheat']['risk_factors']}")
```

### Test Risk Engine Integration:
```python
# Call risk engine endpoint
response = requests.post(
    "http://localhost:8000/evaluate/risk-engine", 
    json=evaluation_data
)

risk_data = response.json()
print(f"LLM Score: {risk_data['llm_score']}")
print(f"Risk Flag: {risk_data['risk_flag']}")
print(f"Cheat Probability: {risk_data['cheat_probability']}")
```

---

## 🎯 API Usage Examples

### 1. Comprehensive Evaluation
```bash
curl -X POST "http://localhost:8000/evaluate/comprehensive" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is polymorphism?",
    "candidate_answer": "Polymorphism allows objects of different types to be treated uniformly through a common interface.",
    "expected_keywords": ["polymorphism", "objects", "interface", "types"],
    "experience_level": "fresher",
    "question_type": "technical"
  }'
```

### 2. Risk Engine Integration
```bash
curl -X POST "http://localhost:8000/evaluate/risk-engine" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain database normalization.",
    "candidate_answer": "Database normalization reduces redundancy by organizing data into tables with relationships.",
    "expected_keywords": ["normalization", "redundancy", "tables", "relationships"],
    "experience_level": "advanced",
    "question_type": "technical"
  }'
```

### 3. Legacy Support (Backward Compatibility)
```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is OOP?",
    "answer": "Object-oriented programming uses classes and objects.",
    "experience_level": "intermediate"
  }'
```

---

## 📊 Understanding Results

### Scoring Breakdown:
- **Technical Accuracy (40%)**: Factual correctness
- **Concept Clarity (25%)**: Clear explanation
- **Keyword Coverage (20%)**: Expected terms used
- **Communication (15%)**: Expression quality

### Anti-Cheating Indicators:
- `is_copy_paste`: Textbook/reference detected
- `is_ai_generated`: AI tool usage detected  
- `is_too_robotic`: Unnatural response patterns
- `transcript_mismatch`: Speech vs. text inconsistency

### Risk Assessment:
- `risk_flag`: True if cheating suspected
- `cheat_probability`: 0-1 confidence score
- `confidence_level`: low/medium/high evaluator confidence

---

## 🔧 Configuration Options

### Custom Scoring Weights:
```python
from app.schemas import ScoringWeights
from app.services.llm_evaluator import LLMEvaluator

# Custom weights
custom_weights = ScoringWeights(
    technical_accuracy=0.50,  # Increase technical weight
    concept_clarity=0.30,
    keyword_coverage=0.10, 
    communication=0.10
)

evaluator = LLMEvaluator(weights=custom_weights)
```

### Mock Mode for Testing:
```bash
export USE_MOCK_MODE=true
# or in .env file:
echo "USE_MOCK_MODE=true" >> .env
```

---

## 📋 Running Comprehensive Tests

```bash
# Run full system test suite
python test_comprehensive_system.py

# Expected output:
# ✅ Architecture test passed
# ✅ Structured output test passed  
# ✅ Scoring rubric test passed
# ✅ Anti-cheating test passed
# ✅ Prompt engineering test passed
# ✅ Error handling test passed
# ✅ Risk engine integration test passed
```

---

## 🌐 API Documentation

Visit `http://localhost:8000/docs` to access interactive API documentation with:
- Complete endpoint descriptions
- Request/response schemas
- Interactive testing interface
- Example requests and responses

---

## 🎯 Integration Ready!

Your system now provides:

✅ **Weighted scoring rubric (40-25-20-15)**  
✅ **Anti-cheating detection**  
✅ **Strict JSON output format**  
✅ **Risk engine integration**  
✅ **Error handling & stability**  
✅ **Backward compatibility**  
✅ **Comprehensive testing**  

The system is ready for integration with your interview monitoring platform!

---

## 🆘 Troubleshooting

### Common Issues:

1. **API Key Error**: 
   - Check `.env` file exists with valid `OPENAI_API_KEY`
   - Set `USE_MOCK_MODE=true` for testing without API

2. **JSON Parsing Error**:
   - System automatically sanitizes responses
   - Fallback to mock evaluation on parsing failure

3. **Low Scores for Good Answers**:
   - Check `expected_keywords` are relevant
   - Verify answer matches `experience_level`
   - Review anti-cheating false positives

4. **High Error Rates**:
   - Monitor API quotas and rate limits
   - Check internet connectivity
   - Enable mock mode for testing

---

Ready to proceed with your next task! 🎉