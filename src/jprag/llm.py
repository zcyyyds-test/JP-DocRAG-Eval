import os
from typing import List, Dict, Any
from src.jprag.config import get_llm_config

try:
    from google import genai
except ImportError:
    genai = None

class LLMGenerator:
    def __init__(self):
        if not genai:
            raise ImportError("Please install google-genai: pip install -q -U google-genai")
            
        self.cfg = get_llm_config()
        self.api_key = self.cfg.get("api_key")
        self.model = self.cfg.get("model", "gemini-3-flash-preview")
        
        # Initialize client (it will use GEMINI_API_KEY env var if api_key is None)
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = genai.Client()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            # Combine system and user prompt for simplicity with basic generate_content
            # (Though 'config' arg supports system_instruction, text concat is robust)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt
            )
            return response.text
        except Exception as e:
            return f"Error generating response: {e}"
