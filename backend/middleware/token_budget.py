from fastapi import HTTPException, status
import tiktoken

class TokenBudgeting:
    def __init__(self, max_tokens_per_request: int = 4000):
        """
        Custom Token Budgeting Class to reject oversized payload requests.
        Protects API Pro limits by ensuring no single request exceeds the budget.
        """
        self.max_tokens = max_tokens_per_request
        # Using a standard fast tokenizer for estimation
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def estimate_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def validate_budget(self, text: str):
        """
        Validates if the text exceeds the predefined token budget.
        Raises an HTTPException if it does.
        """
        token_count = self.estimate_tokens(text)
        if token_count > self.max_tokens:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Request exceeds token budget. Found {token_count} tokens, max allowed is {self.max_tokens}."
            )
        return token_count

# Singleton instance for the app
token_budgeter = TokenBudgeting()
