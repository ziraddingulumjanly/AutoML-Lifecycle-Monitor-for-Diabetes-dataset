from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    # single record or batch
    records: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(..., description="A single record dict or list of record dicts.")

class PredictResponse(BaseModel):
    version: str
    predictions: List[float]
