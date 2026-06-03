import re
from fastapi import HTTPException, status

# llm-guard imports
from llm_guard.input_scanners import PromptInjection, Anonymize
from llm_guard.input_scanners.prompt_injection import MatchType
from llm_guard.vault import Vault

class DualLayerInjectionDefense:
    def __init__(self):
        # Layer 1: Fast regex for obvious malicious prompts
        # E.g., looking for "ignore previous instructions", "system prompt", etc.
        self.malicious_regex = re.compile(
            r"(ignore\s+(all\s+)?previous\s+(instructions|directions|prompts))|"
            r"(you\s+are\s+now\s+(a\s+)?)|"
            r"(system\s+prompt)", 
            re.IGNORECASE
        )
        
        # Layer 2: llm-guard for semantic jailbreaks and PII masking
        self.vault = Vault()
        self.injection_scanner = PromptInjection(match_type=MatchType.FULL)
        self.anonymizer = Anonymize(vault=self.vault)

    def secure_prompt(self, prompt: str) -> str:
        """
        Runs the prompt through a dual-layer defense.
        1. Fast Regex filtering
        2. LLM-Guard semantic scanning and PII anonymization
        Returns the sanitized prompt, or raises an HTTPException.
        """
        # Layer 1
        if self.malicious_regex.search(prompt):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Security violation: Malicious pattern detected in prompt."
            )
            
        # Layer 2 - Anonymize PII
        sanitized_prompt, is_valid, risk_score = self.anonymizer.scan(prompt)
        
        # Layer 2 - Jailbreak Detection
        # PromptInjection scanner
        _, is_safe, inj_risk_score = self.injection_scanner.scan(sanitized_prompt)
        
        if not is_safe:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Security violation: Semantic prompt injection detected."
            )
            
        return sanitized_prompt

# Singleton instance
injection_defense = DualLayerInjectionDefense()
