import re
from datetime import date, datetime
from typing import Any, Type, TypeVar
from pydantic import BaseModel, field_validator, ValidationError

# Local imports
from backend.models.output import AnswerWithEvidence

T = TypeVar("T", bound=BaseModel)

# GSTIN format regex
GSTIN_REGEX = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}$")

# Luhn-mod-36 character map
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def luhn_mod_36_checksum(gstin: str) -> bool:
    if len(gstin) != 15:
        return False
    
    factor = 1
    sum_val = 0
    # Process the first 14 characters
    for char in gstin[:-1]:
        code_point = CHARS.index(char)
        addend = factor * code_point
        # Add the two digits of the base-36 representation of the addend
        sum_val += (addend // 36) + (addend % 36)
        # Alternate factor between 1 and 2
        factor = 2 if factor == 1 else 1
    
    # Calculate checksum character
    remainder = sum_val % 36
    check_code_point = (36 - remainder) % 36
    expected_char = CHARS[check_code_point]
    
    return expected_char == gstin[-1]


class DocumentMetadata(BaseModel):
    gstin: str | None = None
    fiscal_date: date | None = None
    title: str

    @field_validator("gstin")
    @classmethod
    def validate_gstin(cls, v: str | None) -> str | None:
        if not v:
            return v
        v = v.upper().replace(" ", "")
        if not GSTIN_REGEX.match(v):
            raise ValueError(f"Invalid GSTIN format: {v}")
        if not luhn_mod_36_checksum(v):
            raise ValueError(f"Invalid GSTIN checksum: {v}")
        return v

    @field_validator("fiscal_date", mode="before")
    @classmethod
    def parse_fiscal_date(cls, v: Any) -> date | None:
        if not v:
            return None
        if isinstance(v, date):
            return v
        if isinstance(v, str):
            v = v.lower().strip()
            # Normalize robustly
            if v.startswith("fy"):
                # E.g. "FY 24-25" -> Assume end of fiscal year 2025-03-31
                parts = v.replace("fy", "").strip().split("-")
                if len(parts) == 2:
                    try:
                        year = int(parts[1])
                        year = 2000 + year if year < 100 else year
                        return date(year, 3, 31)
                    except ValueError:
                        pass
            try:
                # ISO format parse
                return datetime.fromisoformat(v).date()
            except ValueError:
                # Try strptime fallbacks
                try:
                    return datetime.strptime(v, "%dth %B %Y").date()
                except ValueError:
                    try:
                        return datetime.strptime(v, "%d %b %Y").date()
                    except ValueError:
                        pass
        raise ValueError(f"Unable to parse fiscal_date: {v}")

class GuardResult:
    def __init__(self, validation_passed: bool, validated_output: Any):
        self.validation_passed = validation_passed
        self.validated_output = validated_output

class NativeGuard:
    """
    A lightweight, Python 3.13 compatible replacement for Guardrails AI.
    Handles Pydantic validation and provides a similar interface.
    """
    def __init__(self, output_class: Type[T]):
        self.output_class = output_class

    def validate(self, data: Any) -> GuardResult:
        try:
            # If data is already a model, dump it to dict first
            if isinstance(data, BaseModel):
                data = data.model_dump()
            
            validated = self.output_class(**data)
            return GuardResult(validation_passed=True, validated_output=validated.model_dump())
        except (ValidationError, TypeError, ValueError):
            return GuardResult(validation_passed=False, validated_output=data)

# Create NativeGuard for output validation
output_guard = NativeGuard(output_class=AnswerWithEvidence)

