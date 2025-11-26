import re

def extract_json_string(markdown_text: str) -> str:
   import re

def extract_json_string(markdown_text: str) -> str:
    """
    Extracts JSON string from markdown code blocks.
    Returns the JSON string or empty string if not found.
    """
    # Remove opening ```json or ``` fences
    cleaned = re.sub(r"```(?:json)?", "", markdown_text, flags=re.IGNORECASE)
    # Remove closing ``` fences
    cleaned = cleaned.replace("```", "").strip()

    # Optional: check if it starts with '{' and ends with '}' to look like JSON
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    return ""

