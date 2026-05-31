from typing import Tuple

def convert_normalized_to_absolute(
    bbox: Tuple[int, int, int, int], 
    page_width: float, 
    page_height: float
) -> Tuple[float, float, float, float]:
    """
    Converts Gemini's native normalized bounding box coordinates [ymin, xmin, ymax, xmax] 
    (scaled 0-1000) into absolute UI coordinates [x0, y0, x1, y1] for a given page dimension.
    
    Args:
        bbox: A tuple of (ymin, xmin, ymax, xmax) normalized 0-1000.
        page_width: The width of the PDF page in points or pixels.
        page_height: The height of the PDF page in points or pixels.
        
    Returns:
        A tuple of (x0, y0, x1, y1) representing absolute coordinates.
    """
    ymin, xmin, ymax, xmax = bbox
    
    # Scale from 0-1000 down to 0-1
    normalized_ymin = ymin / 1000.0
    normalized_xmin = xmin / 1000.0
    normalized_ymax = ymax / 1000.0
    normalized_xmax = xmax / 1000.0
    
    # Calculate absolute coordinates
    x0 = normalized_xmin * page_width
    y0 = normalized_ymin * page_height
    x1 = normalized_xmax * page_width
    y1 = normalized_ymax * page_height
    
    return (x0, y0, x1, y1)
