import os
import yaml
from pathlib import Path

def load_config(path: str = "config.yaml"):
    if not os.path.exists(path):
        # Fallback default
        return {
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "api_key_env": "OPENAI_API_KEY"
            }
        }
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_llm_config():
    cfg = load_config()
    llm_cfg = cfg.get("llm", {})
    
    # Resolve API Key
    env_var = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(env_var)
    
    if not api_key:
        print(f"Warning: Environment variable {env_var} not set.")
        
    llm_cfg["api_key"] = api_key
    return llm_cfg
