"""
Test suite for the data module.

Test modules:
- test_models.py: Pydantic models tests
- test_loader.py: Dataset loader tests
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))