from dataclasses import dataclass, asdict
import pandas as pd
from typing import List, Optional


@dataclass
class Profile:
    platform: str
    product: str
    reviews: Optional[pd.DataFrame]
    processed_reviews: Optional[pd.DataFrame] = None
    summary: Optional[List[dict]] = None
    