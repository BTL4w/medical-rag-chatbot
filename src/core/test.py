import yaml
from typing import Dict

def load_prompts(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

prompts = load_prompts("config/prompts.yaml")
print(prompts)