import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class ModelConfig:
    model_name: str
    max_new_tokens: int = 512
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 0.7
    num_return_sequences: int = 1

    @classmethod
    def from_json_file(cls, file_path: str | Path) -> "ModelConfig":
        """Load configuration from a JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)

        return cls(**data)
