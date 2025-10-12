import json
import re
from typing import Optional, Dict, Any
import requests

from ..base import VannaBase
from ..exceptions import DependencyError


class Llama(VannaBase):
    def __init__(self, config=None):
        if not config:
            raise ValueError("config must be provided")
        
        self.mode = config.get('mode', 'local')  # 'local' or 'server'
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 512)
        self.top_p = config.get("top_p", 0.95)
        self.top_k = config.get("top_k", 40)
        self.repeat_penalty = config.get("repeat_penalty", 1.1)
        
        if self.mode == 'local':
            self._init_local_model(config)
        elif self.mode == 'server':
            self._init_server_connection(config)
        else:
            raise ValueError("mode must be either 'local' or 'server'")
    
    def _init_local_model(self, config):
        """Initialize local llama.cpp model"""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method, run command:"
                " \npip install llama-cpp-python"
            )
        
        if 'model_path' not in config:
            raise ValueError("config must contain model_path for local mode")
        
        self.model_path = config["model_path"]
        self.n_ctx = config.get("n_ctx", 2048)
        self.n_threads = config.get("n_threads", 8)  # Mac Studio has many cores
        self.n_gpu_layers = config.get("n_gpu_layers", 1)  # Metal acceleration
        
        # Initialize local model
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=config.get("verbose", False),
            use_mmap=True,  # Memory mapping for efficiency
            use_mlock=config.get("use_mlock", False)  # Lock model in RAM if needed
        )
        
        self.log(f"Initialized local model: {self.model_path}")
    
    def _init_server_connection(self, config):
        """Initialize connection to llama.cpp server"""
        self.server_url = config.get('server_url', 'http://localhost:8080')
        self.api_key = config.get('api_key', None)
        
        # Test connection - try /v1/models or root endpoint
        try:
            response = requests.get(f"{self.server_url}/v1/models", timeout=5)
            if response.status_code == 200:
                self.log(f"Connected to llama.cpp server at {self.server_url}")
            else:
                raise ConnectionError(f"Server returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to llama.cpp server: {e}")
        
        self.llm = None  # No local model in server mode

    def system_message(self, message: str) -> dict:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> dict:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> dict:
        return {"role": "assistant", "content": message}

    def extract_sql(self, llm_response: str) -> str:
        """
        Extracts SQL from LLM response following the same logic as Ollama implementation.
        """
        # Clean up response
        llm_response = llm_response.replace("\\_", "_")
        llm_response = llm_response.replace("\\", "")

        # Try to find SQL in code blocks
        sql = re.search(r"```sql\n((.|\n)*?)(?=;|\[|```)", llm_response, re.DOTALL)
        
        # Try to find SELECT or WITH statements
        select_with = re.search(
            r'(select|(with.*?as \())(.*?)(?=;|\[|```)',
            llm_response,
            re.IGNORECASE | re.DOTALL
        )
        
        if sql:
            self.log(f"Output from LLM: {llm_response} \nExtracted SQL: {sql.group(1)}")
            return sql.group(1).replace("```", "")
        elif select_with:
            self.log(f"Output from LLM: {llm_response} \nExtracted SQL: {select_with.group(0)}")
            return select_with.group(0)
        else:
            return llm_response

    def submit_prompt(self, prompt, **kwargs) -> str:
        self.log(f"Prompt Content:\n{json.dumps(prompt, ensure_ascii=False)}")

        if self.mode == 'server':
            return self._submit_to_server(prompt)
        
        # Local mode (existing code)
        formatted_prompt = self._format_messages(prompt)
        response = self.llm(
            formatted_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["</s>", "\n\n\n"],
            echo=False
        )
        
        response_text = response['choices'][0]['text']
        self.log(f"LlamaCpp Response:\n{response_text}")
        return response_text

    def _submit_to_server(self, prompt) -> str:
        """Submit to llama.cpp server"""
        formatted_prompt = self._format_messages(prompt)
        
        payload = {
            'prompt': formatted_prompt,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'stop': ["</s>", "\n\n\n"]
        }
        
        response = requests.post(
            f"{self.server_url}/completion",
            json=payload,
            timeout=120
        )
        
        result = response.json()
        return result['content']

    def _format_messages(self, messages) -> str:
        """
        Format chat messages into a prompt string for llama.cpp
        """
        formatted = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
        
        # Add final Assistant prompt if last message wasn't from assistant
        if messages and messages[-1].get("role") != "assistant":
            formatted += "Assistant: "
        
        return formatted