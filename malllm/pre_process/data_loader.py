from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class DataSamples:
    name: str
    package_name: str
    package_path: Optional[Path] = None
    package_length: Optional[int] = None
    label: Optional[str] = None

@dataclass
class Dataloader:
    source_dir: str = "../data"


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
                samples.append(DataSamples(
                    name=file_path.name,
                    package_name=file_path.parts[-2],
                    package_path=file_path,
                    label=label,
                    package_length=file_path.stat().st_size
                ))

        return samples