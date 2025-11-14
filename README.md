# MalLLM - LLM-Powered Malicious Code Detection via Code Slicing
This project builds an LLM-powered security analysis system that uses code slicing to automatically detect malicious code inside npm packages by extracting relevant code slices and evaluating them for harmful behavior.
# Workflow Design
<img width="1015" height="758" alt="Screenshot 2025-11-13 at 15 42 58" src="https://github.com/user-attachments/assets/25b947b9-54f4-4354-a16b-76a2d275cd66" />

# Installation
1. Create a vitural environment and activate it
```bash
python3 -m venv venv
# On Window 
.\venv\Scripts\Activate.ps1
# On Macos/linux
source venv/bin/active
```
2. Install dependencies (in venv)
```bash
pip install -r requirements.txt
```
# Usage
1. Create your own config json file to set up your favorite config and model

 For example:
```json
{
  "model_name": "deepseek-ai/deepseek-coder-6.7b-instruct",
  "max_new_tokens": 512,
  "do_sample": true,
  "top_k": 50,
  "top_p": 0.95,
  "temperature": 0.7,
  "num_return_sequences": 1
}
```
2. In file main.py config the path to your json config file
```python
config = ModelConfig.from_json_file("<PATH TO JSON CONFIG FILE>")
```
3. Config slices to detect (Thang will update)


Coming soon ... 

4. Run the model 
```bash
python3 main.py
```
5. Output JSON example:
```
{
  "purpose": string,                 // short single-line description of what this code appears to do
  "sources": [string],               // array of places where input/data is read (e.g., "req.body · line 4", "process.env · line ~1")
  "sinks": [string],                 // array of sensitive sinks or effects (e.g., "eval(...) · line 10", "fs.writeFile · line 12", "exec(...) · line 17")
  "flows": [string],                 // array of source→sink paths with evidence (e.g., "req.body -> eval (line 4 -> line 10)")
  "anomalies": [string],             // unusual patterns, syntax oddities, obfuscation indicators, commented-out dangerous code etc.
  "analysis": string,                // step-by-step numbered analysis of the entire fragment (concise paragraphs)
  "conclusion": string,              // short summary conclusion (one or two sentences)
  "confidence": float,               // overall confidence (0.00-1.00)
  "obfuscated": float,               // estimated obfuscation likelihood (0.00-1.00)
  "malware": float,                  // estimated malware likelihood (0.00-1.00)
  "securityRisk": float              // estimated security risk severity (0.00-1.00)
}
```
# Dataset (Thang and Minh will update)
Comming soon...
# Acknowledgements
In this project we used:
- [JSCodeSlicing](https://github.com/Nguyen-Dang-Khoa-04072004/JSCodeSlicing) tool to slice a package into mutiple code snippets
- The system prompt which created by chatGPT 
- Output format is referenced from the article “Leveraging Large Language Models to Detect npm Malicious Packages.”
