import re
import os
import json
from fastapi import HTTPException, status
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

class GuardResult(BaseModel):
    is_safe: bool = Field(description="True if the prompt is safe, False if it is a semantic jailbreak, prompt injection, or contains severe violations.")
    reason: str = Field(description="Reason for the safety classification.")

class ServerlessInjectionDefense:
    def __init__(self):
        # Layer 1: Fast regex for obvious malicious prompts and PII
        self.malicious_regex = re.compile(
            r"(ignore\s+(all\s+)?previous\s+(instructions|directions|prompts))|"
            r"(you\s+are\s+now\s+(a\s+)?)|"
            r"(system\s+prompt)|"
            r"(\b\d{3}-\d{2}-\d{4}\b)|" # SSN basic pattern
            r"(\b(select|drop|insert|delete|update|union|truncate)\b.*\b(from|into|table)\b)", 
            re.IGNORECASE
        )

    def secure_prompt(self, prompt: str) -> str:
        """
        Runs the prompt through a serverless dual-layer defense.
        1. Fast Regex filtering
        2. API-Driven LLM-Guard semantic scanning (Groq Llama-Guard-3-8B)
        """
        # Layer 1: Regex
        if self.malicious_regex.search(prompt):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Security violation: Malicious pattern or PII detected in prompt."
            )
            
        # Layer 2: Groq Llama-Prompt-Guard-2-86M
        try:
            llm = ChatGroq(
                model="meta-llama/llama-prompt-guard-2-86m", 
                groq_api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.0
            )
            
            # Llama Prompt Guard returns 'safe' or 'unsafe'
            result = llm.invoke(prompt)
            
            if "unsafe" in result.content.lower():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Security violation: Prompt rejected by LLM Guard. {result.content}"
                )
        except HTTPException:
            raise
        except Exception as e:
            if "rate_limit_exceeded" in str(e).lower() or "429" in str(e):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Groq API limit reached. Please try again later."
                )
            # Fail closed for security
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Security validation service is currently unavailable."
            )
            
        return prompt

# Singleton instance
injection_defense = ServerlessInjectionDefense()
