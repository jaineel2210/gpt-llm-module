# 🎯 Prompt Engineering Documentation

## Overview

This document provides comprehensive information about the prompt engineering implementation in the Advanced LLM Interview Evaluation System. It covers prompt design, optimization techniques, and best practices for maintaining consistent, fair evaluations.

---

## 📝 Core Prompt Structure

### Current Production Prompt Template

```
You are an expert AI interview evaluator with advanced anti-cheating detection capabilities.

EVALUATION FRAMEWORK:
Evaluate answers using a weighted 4-parameter scoring system (0-10 scale):

1. TECHNICAL ACCURACY (40% weight)
   - Factual correctness of information
   - Understanding of core concepts
   - Proper use of terminology
   - Accuracy of examples/explanations

2. CONCEPT CLARITY (25% weight)  
   - Clear explanation of concepts
   - Logical flow of ideas
   - Appropriate depth for experience level
   - Well-structured response

3. KEYWORD COVERAGE (20% weight)
   - Coverage of expected keywords: {expected_keywords}
   - Demonstration of domain knowledge
   - Use of relevant technical vocabulary
   - Missing critical concepts

4. COMMUNICATION (15% weight)
   - Clarity and coherence
   - Professional language use
   - Appropriate tone and style
   - Effective explanation skills

ANTI-CHEATING DETECTION:
Analyze for suspicious patterns:

✓ Copy-paste indicators: Textbook language, citations, overly formal tone
✓ AI-generated content: Generic responses, lack of personal insight, robotic patterns
✓ Transcript mismatch: Text doesn't match natural speech patterns (if audio provided)
✓ Unrealistic perfection: Too polished for experience level

EXPERIENCE LEVEL CALIBRATION:
- FRESHER: Basic understanding expected, simple explanations acceptable
- INTERMEDIATE: Good grasp with some examples, moderate depth
- ADVANCED: Deep expertise, comprehensive explanations, industry insights

INPUT DATA:
Question: {question}  
Candidate Answer: {candidate_answer}
Expected Keywords: {expected_keywords}
Experience Level: {experience_level}
Question Type: {question_type}
Context: {context}
Audio Transcript: {audio_transcript}

RESPONSE REQUIREMENTS:
Return STRICTLY VALID JSON with NO additional text. Must include ALL fields below:

{
  "scores": {
    "technical_accuracy": <float 0-10>,
    "concept_clarity": <float 0-10>, 
    "keyword_coverage": <float 0-10>,
    "communication": <float 0-10>,
    "final_score": <calculated weighted average>,
    "confidence_score": <1-10 evaluator confidence>
  },
  "feedback": "<constructive feedback paragraph>",
  "anti_cheat": {
    "is_copy_paste": <boolean>,
    "is_ai_generated": <boolean>,
    "is_too_robotic": <boolean>,
    "transcript_mismatch": <boolean>,  
    "confidence_level": <0-1 float>,
    "risk_factors": [<list of detected issues>]
  },
  "keyword_analysis": {
    <map each expected keyword to boolean coverage>
  },
  "response_quality": "<excellent/good/fair/poor>",
  "areas_for_improvement": [<list of specific suggestions>],
  "processing_metadata": {
    "answer_length": <word count>,
    "complexity_level": "<low/medium/high>",
    "domain_match": <boolean>
  }
}

CRITICAL RULES:
- NO text outside JSON structure
- All scores as numbers (not strings)  
- Boolean values as true/false (not "true"/"false")
- Calculate final_score = (technical_accuracy × 0.4) + (concept_clarity × 0.25) + (keyword_coverage × 0.2) + (communication × 0.15)
- Flag high cheating risk if answer seems copied/AI-generated
- Consider experience level when scoring
```

---

## 🔧 Prompt Optimization Techniques

### Version Control and A/B Testing

We maintain multiple prompt versions for continuous optimization:

1. **Base Version (v1.0)**: Original comprehensive prompt
2. **Concise Version (v1.1)**: Shortened for faster processing
3. **Anti-Hallucination Version (v1.2)**: Enhanced to reduce false information
4. **Consistency Version (v1.3)**: Improved for uniform scoring

### Testing Methodology

```python
from app.services.prompt_optimizer import PromptOptimizer

# Create optimizer
optimizer = PromptOptimizer()

# Add new prompt version
optimizer.add_prompt_version(
    version_id="custom_v1",
    name="Custom Optimized Prompt",
    template=your_prompt_template,
    description="Optimized for specific use case"
)

# Run comparison
report = await optimizer.compare_prompt_versions(
    version_ids=["base_v1", "custom_v1"]
)
```

### Optimization Metrics

We track the following metrics for prompt optimization:

1. **Consistency Score**: Variance in scores across multiple runs
2. **Hallucination Rate**: Frequency of fabricated information
3. **JSON Compliance**: Success rate of valid JSON responses
4. **Response Time**: Average evaluation duration
5. **Score Distribution**: Range and distribution of scores

---

## 📊 Anti-Cheating Prompt Engineering

### Detection Strategies

#### Copy-Paste Detection
```
Indicators to flag:
- "According to textbook", "as mentioned in", "reference:"
- Academic citations and bibliography references
- Overly formal language inconsistent with experience level
- Perfect grammar and structure beyond candidate capability
```

#### AI-Generated Content Detection
```
Patterns to identify:
- "As an AI", "artificial intelligence", "I'm designed to"
- Generic responses lacking personal insight
- Unnatural perfection for stated experience level
- Absence of hesitation markers or filler words
```

#### Robotic Response Detection
```
Check for natural speech patterns:
- Presence of contractions ("don't", "can't", "it's")
- Filler words and hesitations ("um", "you know", "so")
- Conversational connectors ("well", "actually", "basically")
- Appropriate informality for interview context
```

### Anti-Cheat Scoring Logic

```python
# Risk calculation
cheat_indicators = sum([
    is_copy_paste,
    is_ai_generated, 
    is_too_robotic,
    transcript_mismatch
])

cheat_probability = (cheat_indicators / 4) * confidence_level

# Risk flag determination
risk_flag = (
    cheat_probability > 0.5 OR
    excessive_formal_language OR
    answer_quality_mismatch_with_delivery
)
```

---

## 🎛️ Prompt Configuration Parameters

### Configurable Elements

1. **Temperature**: Controls response randomness (default: 0.2)
2. **Max Tokens**: Response length limit (default: 1500)
3. **Top-p**: Nucleus sampling parameter (default: 0.9)
4. **Frequency Penalty**: Repetition reduction (default: 0.1)
5. **Presence Penalty**: Topic diversity (default: 0.1)

### Experience Level Adaptations

#### Fresher Level Adjustments
```
- Accept simpler explanations
- Lower technical depth expectations
- Focus on basic concept understanding
- More lenient communication scoring
```

#### Intermediate Level Requirements
```
- Expect some examples and context
- Moderate technical depth
- Good grasp of fundamentals
- Professional communication
```

#### Advanced Level Expectations
```
- Deep expertise and insights
- Industry best practices
- Comprehensive explanations
- Technical precision
```

---

## 🔄 Prompt Iteration Process

### Development Workflow

1. **Draft Creation**: Initial prompt based on requirements
2. **Internal Testing**: Evaluation against benchmark dataset
3. **A/B Testing**: Comparison with existing versions
4. **Bias Assessment**: Fairness testing across demographics
5. **Production Deployment**: Gradual rollout with monitoring
6. **Continuous Monitoring**: Ongoing performance tracking

### Quality Metrics Tracking

```python
# Key metrics to monitor
metrics = {
    "consistency": target > 0.85,
    "hallucination_rate": target < 0.1,
    "json_compliance": target > 0.95,
    "bias_rate": target < 0.1,
    "avg_response_time": target < 3000  # milliseconds
}
```

---

## ⚠️ Common Pitfalls and Solutions

### Pitfall 1: Inconsistent Scoring
**Problem**: Same answer gets different scores on multiple runs
**Solution**: Add specific scoring rubrics and examples in prompt

### Pitfall 2: Length Bias
**Problem**: Longer answers consistently score higher
**Solution**: Emphasize content quality over quantity

### Pitfall 3: Technical Jargon Bias
**Problem**: Complex vocabulary preferred over clear explanations
**Solution**: Explicitly value clear communication

### Pitfall 4: JSON Formatting Issues
**Problem**: Responses include text outside JSON structure
**Solution**: Multiple enforcement statements and examples

---

## 📈 Performance Optimization

### Response Time Optimization
```python
# Techniques for faster responses
optimizations = {
    "shorter_prompts": "Reduce token count while maintaining clarity",
    "focused_instructions": "Specific rather than general guidance",
    "example_reduction": "Minimal but effective examples",
    "caching_strategies": "Cache common prompt components"
}
```

### Token Usage Management
```python
# Token allocation strategy
token_budget = {
    "system_prompt": ~800_tokens,
    "user_input": ~300_tokens,
    "response_generation": ~1500_tokens,
    "safety_buffer": ~400_tokens
}
```

---

## 🧪 Testing and Validation

### Test Cases for New Prompts

1. **Excellent Answers**: Should score 8.5-10.0
2. **Good Answers**: Should score 7.0-8.4
3. **Average Answers**: Should score 5.0-6.9
4. **Poor Answers**: Should score 2.0-4.9
5. **Bad Answers**: Should score 0.0-1.9

### Validation Checklist

- [ ] JSON format compliance (100% success rate)
- [ ] Score consistency (CV < 0.15)
- [ ] Appropriate score distribution across quality levels
- [ ] Anti-cheat detection accuracy
- [ ] No systematic bias across demographics
- [ ] Response time within acceptable limits

---

## 📋 Maintenance and Updates

### Regular Review Schedule
- **Weekly**: Performance metrics review
- **Monthly**: Bias and fairness assessment
- **Quarterly**: Full prompt optimization cycle
- **Annually**: Complete system architecture review

### Version Documentation
Each prompt version should include:
- Version number and date
- Changes from previous version
- Testing results and metrics
- Rollback procedures
- Performance comparisons

---

## 🔍 Debugging Prompt Issues

### Common Debugging Steps

1. **Check JSON Validity**: Ensure proper format
2. **Review Score Distribution**: Look for outliers
3. **Analyze Consistency**: Multiple runs on same input
4. **Test Edge Cases**: Short answers, long answers, technical jargon
5. **Bias Assessment**: Check across demographics

### Debug Tools

```python
from app.services.prompt_optimizer import PromptOptimizer

# Debug single evaluation
debugger = PromptOptimizer()
result = debugger.debug_single_evaluation(
    prompt_version="current",
    test_input=problematic_input
)

# Compare prompt versions
comparison = await debugger.compare_prompt_versions([
    "current", "previous", "experimental"
])
```

---

This prompt engineering documentation ensures consistent, fair, and optimized evaluation across all interview scenarios while maintaining high-quality standards and bias-free assessments.