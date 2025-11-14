"""
HANDLE MALFORMED JSON FROM LLM RESPONSES
"""

from typing import Dict
import json
import logging
import re
from typing import Any, Optional, Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class JSONParseError(Exception):
    """Custom exception for JSON parsing errors."""
    pass

class LLMResponseParse:

    def __init__(self, max_repair_attempts: int = 3):
        self.max_repair_attempts = max_repair_attempts
        
    def parse_llm_response(
        self,
        response_llm_text: str,
        expected_model: Optional[Type[BaseModel]] = None,
        strict: bool = False
    ) -> Optional[Dict[str, Any] | BaseModel]:
        """
        Robustly parse JSON from LLM response.
        
        Args:
            response_text: Raw LLM output
            expected_model: Pydantic model for validation (optional)
            strict: If True, raise exception; if False, return None
        
        Returns:
            Parsed dict or Pydantic model if expected_model provided, or None on failure
        
        Strategies:
            1. Extract from markdown
            2. Direct JSON parsing
            3. Find JSON object  (consider)
            4. Safe repair json errors (trailing commas, quotes)
            5. Key-value extraction
        """
        
        if not response_llm_text or not response_llm_text.strip():
            return self._handle_error("Empty LLM response text", strict)
        
        parsed_response = self._extract_from_markdown(response_llm_text)
        if parsed_response:
            return self._validate_and_return(parsed_response, expected_model, strict)
        
        parsed_response = self._parse_json_directly(response_llm_text)
        if parsed_response:
            return self._validate_and_return(parsed_response, expected_model, strict)
        
        parsed_response = self._repair_and_parse(response_llm_text)
        if parsed_response:
            return self._validate_and_return(parsed_response, expected_model, strict)
        
        parsed_response = self._extract_key_value_pairs(response_llm_text)
        if parsed_response:
            return self._validate_and_return(parsed_response, expected_model, strict)
        
        logger.error(f"Parsing failed with response: {response_llm_text}")
        return self._handle_error("Parsing response failed", strict)
        
        
    def _extract_from_markdown(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from markdown code blocks."""
        
        patterns = [
            r"```json\s*(\{[\s\S]*?\})\s*```",                              # json
            r"```(?:javascript|js|ts|typescript)\s*(\{[\s\S]*?\})\s*```",   # javascript, js, ts, typescript
            r"```\s*(\{[\s\S]*?\})\s*```",                                  # no label
            r"(\{(?:[^{}]|(?:\{[^{}]*\}))*\})"                              # Inline JSON
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                cleaned_text = match.strip()
                try:
                    return json.loads(cleaned_text)
                except json.JSONDecodeError:
                    res = self._safe_repair_json(cleaned_text)
                    if res:
                        return res
        return None
    
    def _parse_json_directly(self, text: str) -> Optional[Dict[str, Any]]:
        """Direct JSON parsing."""
        try:
            cleaned_text = text.strip()
            if cleaned_text.startswith("{") and cleaned_text.endswith("}"):
                return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parsing failed: {e}")
        return None
    
    def _repair_and_parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Strategy 4: Iterative repair with multiple attempts."""
        for attempt in range(self.max_repair_attempts):
            repaired = self._safe_repair_json(text)
            if repaired:
                return repaired
            
            if attempt > 0:
                text = re.sub(r'^[^{]*', '', text)
                text = re.sub(r'}[^}]*$', '}', text)
        
        return None
    
    def _safe_repair_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Safely repair common JSON issues."""
        res = text
        
        # Remove }, ]
        res = re.sub(r',\s*}', '}', res)
        res = re.sub(r',\s*]', ']', res)
        
        # Add "" for key
        res = re.sub(r'(?<!["\'])\b(\w+)\s*:', r'"\1":',res)
        
        # Replace '' with "" of the key
        res = re.sub(r"'(\w+)'\s*:", r'"\1":', res)
        
        # Replace '' of simple value, not for case apostrophe inside
        res = re.sub(r":\s*'([^']*?(?<!\w)'(?!\w)[^']*?)'\s*([,}])", r': "\1"\2', res)
        
        # Remove comments
        res = re.sub(r'//.*?$', '', res, flags=re.MULTILINE)
        res = re.sub(r'/\*.*?\*/', '', res, flags=re.DOTALL)

        try:
            return json.loads(res)
        except json.JSONDecodeError as e:
            logger.debug(f"Repair failed: {e}")
            return None
        
    
    def _extract_key_value_pairs(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract key-value pairs from text.
        Args:
            text: Raw LLM output
        Returns:
            Dict of extracted key-value pairs or None
        """
        
        patterns = [
            r'"(\w+)"\s*:\s*"([^"]*)"',           # "key": "value"
            r'"(\w+)"\s*:\s*(\d+\.?\d*)',         # "key": integer/float
            r'"(\w+)"\s*:\s*(true|false|null)',   # "key": true/false/null
            r"'(\w+)'\s*:\s*'([^']*)'",           # 'key': 'value'
        ]
        
        result = {}
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for key, value in matches:
                temp_key = key.strip();
                if temp_key not in result:
                    result[temp_key] = self._infer_value_type(value)
        
        return result if result else None

    def _infer_value_type(self, value: str) -> Any:
        """
        Infer appropriate type for extracted values.
        Args:
            value: Extracted string value
        Returns:
            Value converted to inferred Python type
        
        """
        value = value.strip().strip('"\'')
        
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        if value.lower() in ['null', 'none']:
            return None
        
        if not value:
            return ""
        
        try:
            if '.' in value or 'e' in value.lower():
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        if value.startswith('[') and value.endswith(']'):
            try:
                return json.loads(value)
            except:
                pass
        
        return value
                
    def _validate_and_return(
        self,
        data: Dict[str, Any],
        expected_model: Optional[Type[BaseModel]],
        strict: bool
    ) -> Optional[Dict[str, Any] | BaseModel]:
        """Validate data against expected mode."""
        
        if expected_model:
            try:
                valid_model = expected_model(**data)
                logger.debug(f"Validation against model succeeded {expected_model.__name__}")
                return valid_model
            except ValidationError as e:
                logger.warning(
                    f"Pydantic validation against model failed: {expected_model.__name__}",
                    f"Errors: {e.errors()}"
                )
                if strict:
                    raise
                return None
        return data

    def _handle_error(self, message: str, strict: bool) -> None:
        """Handle errors based on strictness."""
        if strict:
            raise JSONParseError(f"LLM response parsing error: {message}")
        return None
    
# ====  UTILITY FUNCTION ====
def parse_llm_response(
    response_llm_text: str,
    expected_model: Optional[Type[BaseModel]] = None,
    strict: bool = False
) -> Optional[Dict[str, Any] | BaseModel]:
    parse = LLMResponseParse()
    
    return parse.parse_llm_response(response_llm_text, expected_model, strict)

