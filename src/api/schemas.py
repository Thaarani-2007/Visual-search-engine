"""
Pydantic Schemas for FastAPI Visual Search Endpoints.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field


class SearchResultItem(BaseModel):
    rank: int = Field(..., description="Retrieval ranking (1 to K)")
    similarity_score: float = Field(..., description="Cosine similarity score in range [-1.0, 1.0]")
    product_id: Optional[Union[int, str]] = Field(default=0, description="Unique product SKU/identity")
    image_id: Optional[Union[int, str]] = Field(default=0, description="Database image identifier")
    relative_path: Optional[str] = Field(default="", description="Relative file path in dataset repository")
    image_url: Optional[str] = Field(default="", description="Direct HTTP URL to display the retrieved image")


class LatencyBreakdown(BaseModel):
    preprocess_ms: float = Field(..., description="Image decoding and transformation time")
    inference_ms: float = Field(..., description="ResNet-50 forward pass time")
    search_ms: float = Field(..., description="FAISS vector retrieval time")
    total_ms: float = Field(..., description="End-to-end request processing time")


class VisualSearchResponse(BaseModel):
    status: str = Field(default="success")
    query_filename: str = Field(..., description="Name of uploaded query image")
    top_k: int = Field(..., description="Number of visual matches retrieved")
    latency: LatencyBreakdown
    results: List[SearchResultItem]


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    index_loaded: bool
    total_indexed_vectors: int
    device: str


class IndexInfoResponse(BaseModel):
    index_type: str
    dimension: int
    total_vectors: int
    metric: str
    hnsw_m: int
    hnsw_ef_search: int
