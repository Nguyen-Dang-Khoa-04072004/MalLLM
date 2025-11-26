import unittest
import json
import sys
from pathlib import Path

# Fix import when script is in tests/ directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from malllm.model.model_config import ModelConfig
from malllm.model.model import AIModel
from malllm.utils.utils import extract_json_string
TEST_DIR = ROOT / "tests-data"

REQUIRED_KEYS = {
    "confidence",
    "obfuscated",
    "malware",
    "securityRisk",
}

def validate_json_structure(data: dict) -> bool:
    """Check JSON structure and values."""
    if not isinstance(data, dict):
        return False
    if set(data.keys()) != REQUIRED_KEYS:
        return False
    for key in REQUIRED_KEYS:
        if not isinstance(data[key], (float, int)):
            return False
    return True


class TestAIModelJSONOutput(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        config_path = ROOT / "config/deepseek-coder-6.7b.json"
        cls.config = ModelConfig.from_json_file(config_path)
        cls.model = AIModel(cls.config)
        assert TEST_DIR.exists(), f"{TEST_DIR} does not exist"

    def test_all_files(self):
        files = list(TEST_DIR.rglob("*"))
        self.assertTrue(files, "No test data files found in tests-data/")

        for file_path in files[0:3]:
            if file_path.is_dir():
                continue

            with self.subTest(file=file_path):
                tokens = self.model.tokenize(file_path)
                raw_output = self.model.generate(tokens)
                print(raw_output)
                raw_output = extract_json_string(raw_output)
                try:
                    data = json.loads(raw_output)
                except json.JSONDecodeError:
                    self.fail(f"Invalid JSON output from {file_path}:\n{raw_output}")

                self.assertTrue(validate_json_structure(data),
                                f"Invalid JSON structure from {file_path}:\n{data}")


if __name__ == "__main__":
    unittest.main()
