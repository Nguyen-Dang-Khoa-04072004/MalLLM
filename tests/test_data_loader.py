"""
Tests loading a few packages from your dataset

"""

import sys
from pathlib import Path

try:
    import data.loader as data_loader
except ImportError:
    sys.exit("Project Failed: Could not import data.loader module.", 1)

try:
    loader = data_loader.Dataloader()

    print(loader.load_data())
except Exception as e:
    sys.exit(f"Project Failed: Could not initialize DataLoader. Error: {e}", 1)

if __name__ == "__main__":
    pass