'''
Function: Generate reponse with a given slice
Note: This script to test generating response by model with a slice. It is not used in workflow
'''
import argparse
from pathlib import Path
from model.model_config import ModelConfig
from model import AIModel


def main(parser_args):
    config = ModelConfig.from_json_file(parser_args.config_file)
    model = AIModel(config)
    file_path = Path("../tests-data/@0x000000000000000#util-0.1.0.txt")
    tokens = model.tokenize(file_path)
    print(model.generate(tokens)) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MalLLM Code Analysis Tool")
    parser.add_argument("--config-file", type=str, default="config/deepseek-coder-6.7b.json", help="File path dẫn đến config file")
    parser.add_argument("--file-path", type=str, default="../tests-data/@0x000000000000000#util-0.1.0.txt")
    parser.add_argument("--max-workers", type=int, default=5, help="Số lượng process worker song song tối đa")
    parser.add_argument("--batch-size", type=int, default=3, help="Số lượng mẫu xử lý trong mỗi batch")
    main(parser.parse_args())