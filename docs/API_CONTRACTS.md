# 📡 API Contract Documentation

## Overview

This document provides comprehensive API specifications for the Advanced LLM Interview Evaluation System. It includes endpoint definitions, request/response schemas, error handling, authentication, and integration examples.

---

## 🚀 Base Configuration

### Server Details
- **Base URL**: `http://localhost:8000/api/v1`
- **Protocol**: HTTP/HTTPS
- **Content-Type**: `application/json`
- **Character Encoding**: UTF-8

### Authentication
```python
# Header-based authentication (when implemented)
headers = {
    "Authorization": "Bearer {api_token}",
    "X-API-Key": "{api_key}",
    "Content-Type": "application/json"
}
```

---

## 📋 Core Endpoints

### 1. Comprehensive Evaluation

**Endpoint**: `POST /evaluate/comprehensive`

**Purpose**: Performs complete LLM evaluation with anti-cheat detection and risk assessment

#### Request Schema

```json
{
  "interview_id": "string (required)",
  "question": "string (required)",
  "answer": "string (required)",
  "experience_level": "string (enum: 'fresher', 'intermediate', 'advanced')",
  "expected_keywords": ["string array (optional)"],
  "position_title": "string (optional)",
  "department": "string (optional)",
  "language": "string (optional, default: 'en')",
  "custom_weights": {
    "technical_accuracy": "number (0.0-1.0, optional)",
    "concept_clarity": "number (0.0-1.0, optional)", 
    "keyword_coverage": "number (0.0-1.0, optional)",
    "communication": "number (0.0-1.0, optional)"
  }
}
```

#### Example Request

```json
{
  "interview_id": "INT_001_20241218",
  "question": "Explain the concept of RESTful APIs and their benefits.",
  "answer": "RESTful APIs are architectural style for designing web services. They use HTTP methods like GET, POST, PUT, DELETE to perform CRUD operations. Benefits include statelessness, scalability, and platform independence.",
  "experience_level": "intermediate",
  "expected_keywords": ["REST", "HTTP", "CRUD", "stateless", "scalability"],
  "position_title": "Backend Developer",
  "department": "Engineering",
  "language": "en"
}
```

#### Response Schema

```json
{
  "status": "string (success/error)",
  "evaluation_id": "string (unique identifier)",
  "overall_score": "number (0.0-10.0)",
  "detailed_scores": {
    "technical_accuracy": "number (0.0-10.0)",
    "concept_clarity": "number (0.0-10.0)",
    "keyword_coverage": "number (0.0-10.0)",
    "communication": "number (0.0-10.0)"
  },
  "weights_used": {
    "technical_accuracy": "number (0.0-1.0)",
    "concept_clarity": "number (0.0-1.0)",
    "keyword_coverage": "number (0.0-1.0)",
    "communication": "number (0.0-1.0)"
  },
  "keyword_analysis": {
    "total_expected": "number",
    "keywords_found": "number",
    "coverage_percentage": "number (0.0-100.0)",
    "missing_keywords": ["string array"],
    "found_keywords": ["string array"],
    "related_terms": ["string array"]
  },
  "anti_cheat": {
    "risk_level": "string (enum: 'low', 'medium', 'high')",
    "confidence": "number (0.0-1.0)",
    "detected_patterns": ["string array"],
    "explanation": "string"
  },
  "feedback": {
    "strengths": ["string array"],
    "areas_for_improvement": ["string array"],
    "recommendations": ["string array"]
  },
  "metadata": {
    "evaluation_time": "string (ISO 8601 timestamp)",
    "processing_duration_ms": "number",
    "model_version": "string",
    "api_version": "string"
  }
}
```

#### Example Response

```json
{
  "status": "success",
  "evaluation_id": "EVAL_001_20241218_143022",
  "overall_score": 7.85,
  "detailed_scores": {
    "technical_accuracy": 8.5,
    "concept_clarity": 7.8,
    "keyword_coverage": 8.0,
    "communication": 7.2
  },
  "weights_used": {
    "technical_accuracy": 0.40,
    "concept_clarity": 0.25,
    "keyword_coverage": 0.20,
    "communication": 0.15
  },
  "keyword_analysis": {
    "total_expected": 5,
    "keywords_found": 4,
    "coverage_percentage": 80.0,
    "missing_keywords": ["platform independence"],
    "found_keywords": ["REST", "HTTP", "CRUD", "stateless"],
    "related_terms": ["web services", "architectural style", "scalability"]
  },
  "anti_cheat": {
    "risk_level": "low",
    "confidence": 0.95,
    "detected_patterns": [],
    "explanation": "Natural response pattern with good technical understanding"
  },
  "feedback": {
    "strengths": [
      "Accurate technical definitions",
      "Good understanding of core concepts",
      "Clear explanation structure"
    ],
    "areas_for_improvement": [
      "Could elaborate on platform independence benefits",
      "More specific examples would strengthen the answer"
    ],
    "recommendations": [
      "Practice explaining benefits with concrete examples",
      "Expand vocabulary around API design patterns"
    ]
  },
  "metadata": {
    "evaluation_time": "2024-12-18T14:30:22.123Z",
    "processing_duration_ms": 2847,
    "model_version": "gpt-4o-mini-2024-07-18",
    "api_version": "v1.0.0"
  }
}
```

---

### 2. Risk Engine Evaluation

**Endpoint**: `POST /evaluate/risk-engine`

**Purpose**: Focused anti-cheat and risk assessment evaluation

#### Request Schema

```json
{
  "interview_id": "string (required)",
  "answer": "string (required)",
  "response_time_ms": "number (optional)",
  "audio_confidence": "number (0.0-1.0, optional)",
  "metadata": {
    "user_agent": "string (optional)",
    "ip_address": "string (optional)",
    "session_duration": "number (optional)"
  }
}
```

#### Example Request

```json
{
  "interview_id": "INT_002_20241218",
  "answer": "Machine learning algorithms learn patterns from data without explicit programming...",
  "response_time_ms": 1500,
  "audio_confidence": 0.87,
  "metadata": {
    "user_agent": "Mozilla/5.0 Chrome/120.0",
    "session_duration": 3600000
  }
}
```

#### Response Schema

```json
{
  "status": "string (success/error)",
  "risk_assessment": {
    "overall_risk": "string (enum: 'low', 'medium', 'high', 'critical')",
    "confidence": "number (0.0-1.0)",
    "risk_score": "number (0.0-100.0)"
  },
  "detected_issues": [
    {
      "type": "string (enum: 'copy_paste', 'ai_generated', 'robotic', 'time_anomaly')",
      "severity": "string (enum: 'low', 'medium', 'high')",
      "confidence": "number (0.0-1.0)",
      "description": "string",
      "evidence": ["string array"]
    }
  ],
  "behavioral_analysis": {
    "response_patterns": ["string array"],
    "timing_analysis": "string",
    "audio_quality_assessment": "string"
  },
  "recommendations": {
    "action": "string (enum: 'accept', 'review', 'reject', 'retest')",
    "reasoning": "string",
    "follow_up_questions": ["string array (optional)"]
  },
  "metadata": {
    "evaluation_time": "string (ISO 8601 timestamp)",
    "processing_duration_ms": "number"
  }
}
```

---

### 3. Batch Evaluation

**Endpoint**: `POST /evaluate/batch`

**Purpose**: Process multiple evaluations in a single request

#### Request Schema

```json
{
  "evaluations": [
    {
      "interview_id": "string (required)",
      "question": "string (required)",
      "answer": "string (required)",
      "experience_level": "string (optional)",
      "expected_keywords": ["string array (optional)"]
    }
  ],
  "options": {
    "include_detailed_feedback": "boolean (default: true)",
    "risk_assessment_only": "boolean (default: false)",
    "async_processing": "boolean (default: false)"
  }
}
```

#### Response Schema

```json
{
  "status": "string (success/partial/error)",
  "batch_id": "string",
  "results": [
    {
      "interview_id": "string",
      "status": "string (success/error)",
      "evaluation": "object (comprehensive evaluation response)",
      "error": "string (if status is error)"
    }
  ],
  "summary": {
    "total_processed": "number",
    "successful": "number",
    "failed": "number",
    "average_score": "number",
    "processing_time_ms": "number"
  }
}
```

---

### 4. Health Check

**Endpoint**: `GET /health`

**Purpose**: System health and status monitoring

#### Response Schema

```json
{
  "status": "string (healthy/degraded/unhealthy)",
  "timestamp": "string (ISO 8601)",
  "version": "string",
  "services": {
    "llm_service": {
      "status": "string",
      "response_time_ms": "number",
      "last_check": "string"
    },
    "database": {
      "status": "string",
      "connection_pool": "number",
      "last_check": "string"
    }
  },
  "metrics": {
    "requests_per_minute": "number",
    "average_response_time": "number",
    "error_rate": "number"
  }
}
```

---

### 5. Configuration

**Endpoint**: `GET /config`

**Purpose**: Retrieve current system configuration

#### Response Schema

```json
{
  "default_weights": {
    "technical_accuracy": "number",
    "concept_clarity": "number", 
    "keyword_coverage": "number",
    "communication": "number"
  },
  "supported_languages": ["string array"],
  "experience_levels": ["string array"],
  "risk_thresholds": {
    "low": "number",
    "medium": "number",
    "high": "number"
  },
  "rate_limits": {
    "requests_per_minute": "number",
    "burst_limit": "number"
  }
}
```

---

## 🚨 Error Handling

### Standard Error Response

```json
{
  "status": "error",
  "error": {
    "code": "string (error code)",
    "message": "string (human-readable message)",
    "details": "object (additional error context)",
    "timestamp": "string (ISO 8601)",
    "request_id": "string (for tracking)"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description | Resolution |
|------|-------------|-------------|------------|
| `INVALID_REQUEST` | 400 | Malformed request body | Check request schema |
| `MISSING_REQUIRED_FIELD` | 400 | Required field not provided | Provide all required fields |
| `INVALID_EXPERIENCE_LEVEL` | 400 | Invalid experience level value | Use: fresher, intermediate, advanced |
| `EVALUATION_FAILED` | 500 | LLM evaluation process failed | Retry request or contact support |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests | Implement rate limiting |
| `SERVICE_UNAVAILABLE` | 503 | LLM service temporarily unavailable | Retry with exponential backoff |

### Example Error Response

```json
{
  "status": "error",
  "error": {
    "code": "MISSING_REQUIRED_FIELD",
    "message": "The 'answer' field is required but was not provided",
    "details": {
      "required_fields": ["interview_id", "question", "answer"],
      "provided_fields": ["interview_id", "question"]
    },
    "timestamp": "2024-12-18T14:30:22.123Z",
    "request_id": "req_12345_67890"
  }
}
```

---

## ⚙️ Rate Limiting

### Default Limits

- **Standard Users**: 100 requests/minute
- **Premium Users**: 500 requests/minute
- **Enterprise**: Configurable limits
- **Burst Allow**: 2x rate for 30 seconds

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1703162400
X-RateLimit-Burst-Remaining: 5
```

### Rate Limit Exceeded Response

```json
{
  "status": "error",
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Limit: 100 requests per minute",
    "details": {
      "limit": 100,
      "period": "minute",
      "retry_after": 45
    }
  }
}
```

---

## 🔐 Security Considerations

### Input Validation

- Request size limits: 10MB maximum
- Answer length limits: 50,000 characters maximum
- Input sanitization against XSS and injection
- Unicode normalization and validation

### Authentication (Future Implementation)

```python
# JWT Token Example
{
  "sub": "user_12345",
  "aud": "llm-evaluation-api",
  "exp": 1703245200,
  "iat": 1703158800,
  "scope": ["evaluate", "batch_process"]
}
```

### CORS Configuration

```http
Access-Control-Allow-Origin: https://your-domain.com
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization
Access-Control-Max-Age: 3600
```

---

## 📊 Integration Examples

### Python Client

```python
import requests
import json

class LLMEvaluationClient:
    def __init__(self, base_url="http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def evaluate_answer(self, interview_id, question, answer, 
                       experience_level="intermediate", expected_keywords=None):
        """Evaluate a single interview answer"""
        endpoint = f"{self.base_url}/evaluate/comprehensive"
        
        payload = {
            "interview_id": interview_id,
            "question": question,
            "answer": answer,
            "experience_level": experience_level,
            "expected_keywords": expected_keywords or []
        }
        
        try:
            response = self.session.post(endpoint, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "error": str(e)}
    
    def batch_evaluate(self, evaluations):
        """Process multiple evaluations"""
        endpoint = f"{self.base_url}/evaluate/batch"
        
        payload = {
            "evaluations": evaluations,
            "options": {
                "include_detailed_feedback": True,
                "async_processing": False
            }
        }
        
        response = self.session.post(endpoint, json=payload)
        return response.json()

# Usage example
client = LLMEvaluationClient()

result = client.evaluate_answer(
    interview_id="INT_001",
    question="What is machine learning?",
    answer="Machine learning is a subset of AI that enables computers to learn patterns from data without explicit programming.",
    experience_level="fresher",
    expected_keywords=["machine learning", "AI", "patterns", "data"]
)

print(f"Overall Score: {result['overall_score']}")
print(f"Risk Level: {result['anti_cheat']['risk_level']}")
```

### JavaScript/Node.js Client

```javascript
class LLMEvaluationAPI {
    constructor(baseUrl = 'http://localhost:8000/api/v1') {
        this.baseUrl = baseUrl;
    }

    async evaluateAnswer(payload) {
        try {
            const response = await fetch(`${this.baseUrl}/evaluate/comprehensive`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Evaluation failed:', error);
            return { status: 'error', error: error.message };
        }
    }

    async checkHealth() {
        try {
            const response = await fetch(`${this.baseUrl}/health`);
            return await response.json();
        } catch (error) {
            return { status: 'unhealthy', error: error.message };
        }
    }
}

// Usage
const api = new LLMEvaluationAPI();

const evaluation = await api.evaluateAnswer({
    interview_id: 'INT_002',
    question: 'Explain REST APIs',
    answer: 'REST APIs are...',
    experience_level: 'intermediate',
    expected_keywords: ['REST', 'HTTP', 'API']
});

console.log('Evaluation Result:', evaluation);
```

### cURL Examples

```bash
# Basic evaluation
curl -X POST "http://localhost:8000/api/v1/evaluate/comprehensive" \
     -H "Content-Type: application/json" \
     -d '{
       "interview_id": "INT_001",
       "question": "What is Docker?",
       "answer": "Docker is a containerization platform that packages applications with their dependencies.",
       "experience_level": "intermediate",
       "expected_keywords": ["Docker", "containerization", "dependencies"]
     }'

# Risk assessment only
curl -X POST "http://localhost:8000/api/v1/evaluate/risk-engine" \
     -H "Content-Type: application/json" \
     -d '{
       "interview_id": "INT_002",
       "answer": "Machine learning is a subset of artificial intelligence...",
       "response_time_ms": 2000,
       "audio_confidence": 0.85
     }'

# Health check
curl -X GET "http://localhost:8000/api/v1/health"

# Configuration
curl -X GET "http://localhost:8000/api/v1/config"
```

---

## 🔄 Webhook Support (Future Feature)

### Webhook Configuration

```json
{
  "webhook_url": "https://your-domain.com/webhooks/evaluation",
  "events": ["evaluation.completed", "risk.detected"],
  "secret": "webhook_secret_key",
  "retry_policy": {
    "max_attempts": 3,
    "backoff_multiplier": 2
  }
}
```

### Webhook Payload Example

```json
{
  "event": "evaluation.completed",
  "timestamp": "2024-12-18T14:30:22.123Z",
  "data": {
    "interview_id": "INT_001",
    "evaluation_id": "EVAL_001_20241218",
    "overall_score": 7.85,
    "risk_level": "low"
  },
  "signature": "sha256=signature_hash"
}
```

---

## 📈 Performance Specifications

### Response Times (95th percentile)

- **Single Evaluation**: < 3 seconds
- **Batch Evaluation (10 items)**: < 15 seconds
- **Risk Assessment**: < 1 second
- **Health Check**: < 100ms

### Throughput Capabilities

- **Concurrent Evaluations**: 50 simultaneous requests
- **Daily Throughput**: 100,000+ evaluations
- **Batch Processing**: Up to 100 items per batch

### Reliability Metrics

- **Uptime SLA**: 99.9%
- **Error Rate**: < 0.1%
- **Data Accuracy**: 97%+ correlation with expert ratings

---

## 🛠️ SDK and Libraries

### Official SDKs (Planned)

- **Python SDK**: `pip install llm-evaluation-sdk`
- **Node.js SDK**: `npm install @company/llm-evaluation`
- **Java SDK**: Maven/Gradle integration
- **Go SDK**: Native Go client library

### Community Libraries

- REST client wrappers for various languages
- Integration plugins for popular HR platforms
- Webhook handling utilities

---

This API contract provides comprehensive specifications for integrating with the Advanced LLM Interview Evaluation System, ensuring reliable and consistent communication between client applications and the evaluation service.