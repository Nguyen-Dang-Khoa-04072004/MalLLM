'''
Function: Generate reponse with a given slice
Note: This script to test generating response by model with a slice. It is not used in workflow
'''
import argparse
from pathlib import Path
from model.model_config import ModelConfig
from model.model import AIModel
from utils.utils import extract_json_string
def main(parser_args):
    config = ModelConfig.from_json_file(parser_args.config_file)
    model = AIModel(config)
    tokens = model.tokenize(Path(parser_args.file_path))
    response = model.generate(tokens)
    print(extract_json_string(response))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MalLLM Code Analysis Tool")
    parser.add_argument("--config-file", type=str, default="../config/qwen-coder-0.5b.json", help="File path dẫn đến config file")
    parser.add_argument("--file-path", type=str, default="../tests-data/@att-bit#duc.components.cardshell-10.0.4.txt")
    parser.add_argument("--max-workers", type=int, default=5, help="Số lượng process worker song song tối đa")
    parser.add_argument("--batch-size", type=int, default=3, help="Số lượng mẫu xử lý trong mỗi batch")
    main(parser.parse_args())
