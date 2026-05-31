from pydantic import BaseModel, Field
from typing import List

class Citation(BaseModel):
    exact_quote: str = Field(..., description="The exact verbatim string quote from the document.")
    bounding_box: List[int] = Field(..., description="The exact [ymin, xmin, ymax, xmax] coordinates from the source.")
    page_number: int | None = Field(None, description="The page number where the quote is located, if available.")

class AnswerWithEvidence(BaseModel):
    answer: str = Field(..., description="The final synthesized answer to the user's question.")
    reasoning: str = Field(..., description="A step-by-step explanation of how the answer was derived from the sources.")
    citations: List[Citation] = Field(default_factory=list, description="List of exact citations supporting the answer.")
