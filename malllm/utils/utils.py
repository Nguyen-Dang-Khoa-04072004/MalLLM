import re
import os
from pathlib import Path
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


def get_packages(root : Path):
    packages = []

    if not os.path.isdir(root):
        print(f"Directory not found: {root}")
        return []

    # List only directories (packages)
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            packages.append(name)

    return packages

