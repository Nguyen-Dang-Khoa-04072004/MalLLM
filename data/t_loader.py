from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class DataSamples:
    file_path: str
    label: Optional[str] = None

@dataclass
class Dataloader:
    source_dir: str = "./data/samples/"


    def load_data(self) -> List[DataSamples]:
        '''
        Load data from the source directory
        '''

        source_path = Path(self.source_dir)
        if not source_path.is_dir():
            print(f"Source directory {self.source_dir} does not exist.")
            return []
        
        malicious_path = source_path / "malicious"
        benign_path = source_path / "benign"

        samples: List[DataSamples] = []

        for category_path, label in [(malicious_path, "malicious"), (benign_path, "benign")]:
            for file_path in category_path.glob("**/*.txt"):
                print(f"Processing file: {file_path}")

                samples.append(DataSamples(file_path=str(file_path), label=label))

        return samples