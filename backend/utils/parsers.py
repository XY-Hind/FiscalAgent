import json
from typing import List, Dict, Any

def encode_to_toon_table(table_name: str, rows_data: List[Dict[str, Any]]) -> str:
    """
    Encodes tabular data into a TOON (Text-Oriented Object Notation) string format
    which is optimized for LLM tokenization and semantic understanding.
    """
    if not rows_data:
        return f"table: {table_name} [empty]"
    
    # Extract headers if the dicts have consistent keys, otherwise just dump them
    lines = [f"table: {table_name}"]
    
    for idx, row in enumerate(rows_data):
        row_str = " | ".join(f"{k}: {v}" for k, v in row.items())
        lines.append(f"  row_{idx}: {row_str}")
        
    return "\n".join(lines)
