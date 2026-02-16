# GPT-LLM Module: Comprehensive Interview Evaluation System

A sophisticated Python-based interview evaluation system that leverages Large Language Models (LLMs) to provide fair, unbiased, and comprehensive assessment of interview responses with advanced features including plagiarism detection, bias testing, and multi-turn context management.

## 🚀 Features

### Core Evaluation
- **Intelligent Scoring**: Multi-dimensional evaluation with technical accuracy, communication skills, and problem-solving assessment
- **JSON-Enforced Responses**: Guaranteed structured output with automatic format validation
- **Confidence Scoring**: Built-in confidence assessment for reliability indicators
- **Contextual Understanding**: Advanced prompt engineering for accurate evaluation

### Advanced Capabilities
- **🛡️ Bias Detection & Mitigation**: Comprehensive bias testing across gender, race, age, and cultural dimensions
- **🔍 Plagiarism Detection**: Multiple detection methods including semantic similarity and pattern matching
- **🔄 Multi-Turn Context Management**: Sophisticated conversation flow management for interview scenarios
- **📊 Benchmark Creation**: Dynamic benchmark generation for consistent evaluation standards
- **⚡ Confidence Adjustment**: Automatic confidence calibration based on response patterns
- **🎯 Prompt Optimization**: Advanced prompt engineering with A/B testing capabilities

### API & Integration
- **FastAPI Framework**: High-performance REST API with automatic documentation
- **Comprehensive Validation**: Input/output validation with detailed error handling
- **Extensible Architecture**: Modular design for easy feature additions
- **Production Ready**: Robust error handling and logging

## 📋 Requirements

- Python 3.8+
- OpenAI API Key
- Internet connection for LLM API calls

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/jaineel2210/gpt-llm-module.git
cd gpt-llm-module
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=1500
OPENAI_TEMPERATURE=0.3
```

### 5. Run the Application
```bash
# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## 🔧 Configuration

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `OPENAI_MODEL` | GPT model to use | `gpt-4` |
| `OPENAI_MAX_TOKENS` | Maximum tokens per response | `1500` |
| `OPENAI_TEMPERATURE` | Model temperature (0-1) | `0.3` |

### Custom Configuration
Edit `app/config.py` to modify:
- Scoring weights and criteria
- Bias detection parameters
- Plagiarism detection thresholds
- Confidence adjustment settings

## 📖 Quick Start

### Basic Evaluation
```python
import requests

# Basic interview evaluation
response = requests.post("http://localhost:8000/evaluate", json={
    "question": "Explain the concept of machine learning",
    "answer": "Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
    "job_role": "Data Scientist"
})

print(response.json())
```

### Advanced Features
```python
# Bias testing
bias_response = requests.post("http://localhost:8000/test-bias", json={
    "question": "Describe your leadership experience",
    "answer": "I led a team of 10 developers on multiple projects",
    "bias_types": ["gender", "age"]
})

# Plagiarism detection
plagiarism_response = requests.post("http://localhost:8000/check-plagiarism", json={
    "answer": "Machine learning is a method of data analysis...",
    "reference_texts": ["Machine learning is a method of data analysis that automates analytical model building"]
})
```

## 🌐 API Documentation

### Main Endpoints

#### `POST /evaluate`
Evaluate an interview response with comprehensive scoring.

**Request Body:**
```json
{
    "question": "string",
    "answer": "string", 
    "job_role": "string",
    "context": "string (optional)"
}
```

**Response:**
```json
{
    "overall_score": 85,
    "technical_accuracy": 90,
    "communication_skills": 80,
    "problem_solving": 85,
    "confidence_score": 92,
    "detailed_feedback": "...",
    "improvement_suggestions": ["..."],
    "bias_indicators": {...},
    "plagiarism_check": {...}
}
```

#### `POST /test-bias`
Test responses for various types of bias.

#### `POST /check-plagiarism` 
Check for plagiarism in responses.

#### `POST /manage-context`
Manage multi-turn conversation context.

#### `GET /docs`
Interactive API documentation (Swagger UI)

### Complete API Reference
Visit `http://localhost:8000/docs` when the server is running for full interactive documentation.

## 🧪 Testing

### Run All Tests
```bash
# Run comprehensive test suite
python -m pytest tests/ -v

# Run specific test modules
python test_comprehensive_system.py
python test_bias_testing.py  
python test_plagiarism_detection.py
python test_confidence_adjustment.py
python test_multi_turn_context.py
```

### Quick Test
```bash
# Quick functionality test
python quick_test.py
```

### Test Coverage
- Unit tests for all core modules
- Integration tests for API endpoints
- Bias testing validation
- Plagiarism detection accuracy
- Context management functionality
- Confidence calibration tests

## 📁 Project Structure

```
gpt_llm_module/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── schemas.py           # Pydantic models
│   ├── services/            # Core business logic
│   │   ├── llm_evaluator.py       # Main evaluation engine
│   │   ├── bias_tester.py          # Bias detection system
│   │   ├── plagiarism_detector.py  # Plagiarism detection
│   │   ├── confidence_adjuster.py  # Confidence calibration
│   │   ├── context_manager.py      # Multi-turn management
│   │   ├── prompt_optimizer.py     # Prompt optimization
│   │   ├── benchmark_creator.py    # Benchmark generation
│   │   └── json_enforcer.py        # JSON response validation
│   ├── utils/
│   │   └── validator.py     # Input validation utilities
│   └── prompts/
│       └── evaluation_prompt.txt   # Evaluation prompts
├── tests/                   # Test modules
├── docs/                    # Documentation
│   ├── API_CONTRACTS.md     # API specifications
│   ├── PROMPT_ENGINEERING.md # Prompt design guide
│   └── SCORING_SYSTEM.md    # Scoring methodology
├── requirements.txt         # Python dependencies
├── QUICK_START.md          # Quick start guide
├── TESTING_GUIDE.md        # Testing instructions
└── README.md               # This file
```

## 📚 Documentation

- **[Quick Start Guide](QUICK_START.md)** - Get started in 5 minutes
- **[API Contracts](docs/API_CONTRACTS.md)** - Detailed API specifications
- **[System Architecture](SYSTEM_ARCHITECTURE.md)** - Technical architecture overview
- **[Prompt Engineering](docs/PROMPT_ENGINEERING.md)** - Prompt design principles
- **[Scoring System](docs/SCORING_SYSTEM.md)** - Evaluation methodology
- **[Testing Guide](TESTING_GUIDE.md)** - Comprehensive testing instructions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`python -m pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

**API Key Errors**
- Ensure your OpenAI API key is correctly set in the `.env` file
- Verify the API key has sufficient credits and permissions

**Installation Issues**
- Use Python 3.8+ 
- Update pip: `python -m pip install --upgrade pip`
- Try installing in a clean virtual environment

**Performance Issues**
- Adjust `OPENAI_MAX_TOKENS` for faster responses
- Increase `OPENAI_TEMPERATURE` for more creative but potentially less consistent results

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/jaineel2210/gpt-llm-module/issues)
- **Documentation**: Check the `docs/` folder for detailed guides

---

## 🏗️ Built With

- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern Python web framework
- **[OpenAI GPT](https://openai.com/)** - Large Language Model
- **[Pydantic](https://pydantic-docs.helpmanual.io/)** - Data validation
- **scikit-learn** - Machine learning utilities
- **pytest** - Testing framework

---

**Made with ❤️ for better interview evaluations**
