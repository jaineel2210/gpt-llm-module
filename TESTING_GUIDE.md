# Testing Guide for GPT LLM Module

## 🚀 Quick Start

### 1. Start the Server

```bash
# Using conda environment (recommended)
conda run -p "C:\Users\JAINEEL PANDYA\anaconda3" --no-capture-output python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Or if uvicorn is in your PATH
uvicorn app.main:app --reload --port 8000
```

Server will run on: **http://127.0.0.1:8000**

---

## 📋 Testing Methods

### Method 1: Interactive API Documentation (Easiest)

1. Open your browser to: **http://127.0.0.1:8000/docs**
2. Click on `POST /evaluate` endpoint
3. Click **"Try it out"** button
4. Use this sample request:

```json
{
  "question": "What is machine learning?",
  "answer": "Machine learning is a subset of AI that allows systems to learn from data and improve automatically without being explicitly programmed.",
  "experience_level": "fresher"
}
```

5. Click **"Execute"**
6. See the evaluation results below

---

### Method 2: Automated Test Script (Comprehensive)

Run the comprehensive test script:

```bash
# Install requests if needed
pip install requests

# Run the test script
python test_api.py
```

This will test:
- ✅ Server health check
- ✅ Multiple evaluation scenarios (fresher, intermediate, advanced)
- ✅ Error handling with invalid requests
- ✅ Different question types

---

### Method 3: Unit Tests

Run unit tests for individual components:

```bash
# Run with pytest (recommended)
pytest test_unit.py -v

# Or run manually
python test_unit.py
```

This tests:
- ✅ JSON validator functionality
- ✅ Pydantic schema validation
- ✅ Error handling for invalid data

---

### Method 4: Using cURL (Command Line)

```bash
curl -X POST "http://127.0.0.1:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is Python?\", \"answer\": \"Python is a programming language\", \"experience_level\": \"fresher\"}"
```

---

### Method 5: Using PowerShell

```powershell
$body = @{
    question = "What is machine learning?"
    answer = "ML is a subset of AI"
    experience_level = "fresher"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/evaluate" -Method POST -Body $body -ContentType "application/json"
```

---

### Method 6: Using Postman

1. Create a new POST request
2. URL: `http://127.0.0.1:8000/evaluate`
3. Headers: `Content-Type: application/json`
4. Body (raw JSON):
```json
{
  "question": "Explain REST API",
  "answer": "REST is an architectural style for web services using HTTP methods",
  "experience_level": "intermediate"
}
```

---

## 🧪 Test Cases to Try

### Test Case 1: Fresher Level - Short Answer
```json
{
  "question": "What is a variable?",
  "answer": "A variable stores data values",
  "experience_level": "fresher"
}
```

### Test Case 2: Intermediate Level - Detailed Answer
```json
{
  "question": "Explain the difference between SQL and NoSQL databases",
  "answer": "SQL databases are relational and use structured schema with tables. NoSQL databases are non-relational and can store unstructured data in various formats like documents, key-value pairs, or graphs. SQL is better for complex queries while NoSQL offers better scalability.",
  "experience_level": "intermediate"
}
```

### Test Case 3: Advanced Level - System Design
```json
{
  "question": "How would you design a distributed cache system?",
  "answer": "I would use consistent hashing for data distribution across nodes, implement cache invalidation strategies like LRU, use replication for fault tolerance, and add a proxy layer for routing. The system would use Redis or Memcached as the underlying cache store with health monitoring.",
  "experience_level": "advanced"
}
```

### Test Case 4: Error Testing - Missing Field
```json
{
  "question": "What is AI?",
  "answer": "Artificial Intelligence"
}
```
**Expected:** 422 Unprocessable Entity error

---

## 📊 Expected Response Format

```json
{
  "relevance_score": 8.5,
  "clarity_score": 7.0,
  "technical_accuracy": 8.0,
  "communication_score": 7.5,
  "overall_score": 7.8,
  "feedback": "The answer demonstrates good understanding..."
}
```

---

## 🔧 Troubleshooting

### Server won't start
- Check if port 8000 is already in use
- Try a different port: `--port 8001`
- Ensure all dependencies are installed: `pip install -r requirements.txt`

### Getting 422 errors
- Verify all three fields are present: `question`, `answer`, `experience_level`
- Check JSON formatting is valid
- Ensure Content-Type header is set to `application/json`

### Mock mode responses
- If `USE_MOCK_MODE=true` in `.env`, you'll get simulated responses
- To use real OpenAI API, set `USE_MOCK_MODE=false` and add valid API key

---

## ✅ What Success Looks Like

- Server starts without errors
- `/docs` page loads successfully
- API returns evaluation scores (0-10 range)
- All required fields present in response
- Feedback text is meaningful and relevant
- Invalid requests return proper error codes

---

## 📝 Quick Test Checklist

- [ ] Server starts successfully
- [ ] API documentation accessible at /docs
- [ ] POST request returns 200 status
- [ ] Response contains all score fields
- [ ] Scores are in valid range (0-10)
- [ ] Feedback text is present
- [ ] Invalid requests return 422 errors
- [ ] Unit tests pass
- [ ] Comprehensive test script completes

---

## 🎯 Performance Testing

For load testing, use:

```bash
# Install Apache Bench or use other tools
ab -n 100 -c 10 -p test_payload.json -T application/json http://127.0.0.1:8000/evaluate
```

---

## 📞 Need Help?

- Check server logs in the terminal where you started uvicorn
- Review error messages in the API response
- Verify `.env` file configuration
- Ensure Python environment has all dependencies installed
