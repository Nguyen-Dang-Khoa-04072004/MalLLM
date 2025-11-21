from pathlib import Path
from typing import Dict, List

class Prompt:
    """Load, manage, and combine text prompts for chat-based LLMs."""

    def __init__(self, file_path: str | Path, role: str = "user"):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.file_path}")

        self.role = role
        self.content = self._load_prompt()

    def _load_prompt(self) -> str:
        """Read and clean prompt from file."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def to_prompt(self) -> List[Dict[str, str]]:
        """
        Return a list of one dict for chat-based models.
        Example: [{"role": "user", "content": "..."}]
        """
        return [{"role": self.role, "content": self.content}]

    @staticmethod
    def combine(*prompts: "Prompt") -> List[Dict[str, str]]:
        """
        Combine multiple Prompt objects into a single list of messages.
        Useful for multi-turn or role-based prompts.
        """
        combined = []
        for p in prompts:
            combined.extend(p.to_prompt())
        return combined

    def __repr__(self):
        return f"<Prompt role='{self.role}' file='{self.file_path.name}' length={len(self.content)}>"
