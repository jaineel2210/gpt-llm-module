"""
Unit tests for individual components of the LLM evaluation module.
Run with: pytest test_unit.py -v
"""
import pytest
from app.utils.validator import validate_llm_output
from app.schemas import EvaluationInput, EvaluationOutput
from pydantic import ValidationError

class TestValidator:
    """Test the LLM output validator"""
    
    def test_valid_output(self):
        """Test with valid JSON output"""
        valid_output = """
        {
            "relevance_score": 8.5,
            "clarity_score": 7.0,
            "technical_accuracy": 8.0,
            "communication_score": 7.5,
            "overall_score": 7.8,
            "feedback": "Good understanding of the concept."
        }
        """
        result = validate_llm_output(valid_output)
        assert result is not None
        assert result["relevance_score"] == 8.5
        assert result["overall_score"] == 7.8
        print("✅ Valid output test passed")
    
    def test_missing_fields(self):
        """Test with missing required fields"""
        invalid_output = """
        {
            "relevance_score": 8.5,
            "clarity_score": 7.0
        }
        """
        result = validate_llm_output(invalid_output)
        assert result is None
        print("✅ Missing fields test passed")
    
    def test_invalid_json(self):
        """Test with invalid JSON"""
        invalid_output = "This is not JSON"
        result = validate_llm_output(invalid_output)
        assert result is None
        print("✅ Invalid JSON test passed")
    
    def test_extra_fields_allowed(self):
        """Test that extra fields don't break validation"""
        output_with_extra = """
        {
            "relevance_score": 8.5,
            "clarity_score": 7.0,
            "technical_accuracy": 8.0,
            "communication_score": 7.5,
            "overall_score": 7.8,
            "feedback": "Good answer",
            "extra_field": "This should be ignored"
        }
        """
        result = validate_llm_output(output_with_extra)
        assert result is not None
        print("✅ Extra fields test passed")


class TestSchemas:
    """Test Pydantic schemas"""
    
    def test_evaluation_input_valid(self):
        """Test valid EvaluationInput"""
        data = {
            "question": "What is Python?",
            "answer": "Python is a programming language.",
            "experience_level": "fresher"
        }
        eval_input = EvaluationInput(**data)
        assert eval_input.question == "What is Python?"
        assert eval_input.experience_level == "fresher"
        print("✅ Valid input schema test passed")
    
    def test_evaluation_input_missing_field(self):
        """Test EvaluationInput with missing field"""
        data = {
            "question": "What is Python?",
            "answer": "Python is a programming language."
            # missing experience_level
        }
        with pytest.raises(ValidationError):
            EvaluationInput(**data)
        print("✅ Missing field validation test passed")
    
    def test_evaluation_output_valid(self):
        """Test valid EvaluationOutput"""
        data = {
            "relevance_score": 8.5,
            "clarity_score": 7.0,
            "technical_accuracy": 8.0,
            "communication_score": 7.5,
            "overall_score": 7.8,
            "feedback": "Good answer"
        }
        eval_output = EvaluationOutput(**data)
        assert eval_output.overall_score == 7.8
        print("✅ Valid output schema test passed")


def run_all_tests():
    """Run all tests manually (without pytest)"""
    print("\n" + "="*60)
    print("🧪 Running Unit Tests")
    print("="*60)
    
    # Test Validator
    print("\n--- Testing Validator ---")
    validator_tests = TestValidator()
    validator_tests.test_valid_output()
    validator_tests.test_missing_fields()
    validator_tests.test_invalid_json()
    validator_tests.test_extra_fields_allowed()
    
    # Test Schemas
    print("\n--- Testing Schemas ---")
    schema_tests = TestSchemas()
    schema_tests.test_evaluation_input_valid()
    schema_tests.test_evaluation_input_missing_field()
    schema_tests.test_evaluation_output_valid()
    
    print("\n" + "="*60)
    print("✅ All unit tests passed!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
