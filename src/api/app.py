"""
Production FastAPI Visual Search Service.

Endpoints:
- POST /search: Uploads query image, computes 512-D embedding, retrieves top-5 matches via FAISS.
- GET /health: Health check and hardware diagnostics.
- GET /index-info: Vector index statistics (dimensions, total vectors, HNSW params).
- GET /: Visual Search Web Interface.
"""

from contextlib import asynccontextmanager
import io
import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from PIL import Image
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import (
    HealthResponse,
    IndexInfoResponse,
    LatencyBreakdown,
    SearchResultItem,
    VisualSearchResponse,
)
from src.config import PROJECT_ROOT, load_config
from src.data.dataset import get_transforms
from src.indexing.faiss_index import FAISSIndexManager
from src.models.siamese_net import ResNet50MetricNet


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan Context Manager.
    Loads ResNet-50 and FAISS vector index strictly ONCE on startup.
    """
    cfg = load_config()
    app.state.cfg = cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.device = device
    print(f"[Startup] Initializing Visual Search Service on device: {device}")

    # 1. Load Trained Metric Learning Model
    ckpt_path = PROJECT_ROOT / cfg["training"]["checkpoint_dir"] / cfg["training"]["best_model_name"]
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}. Train the model first.")

    model = ResNet50MetricNet(
        embedding_dim=cfg["model"]["embedding_dim"],
        pretrained=False,
        freeze_early_layers=False,
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    app.state.model = model
    print(f"[Startup] ResNet-50 Metric Model loaded successfully from {ckpt_path}")

    # 2. Preprocessing Transformation Pipeline
    app.state.transform = get_transforms(split="test", image_size=cfg["preprocessing"]["image_size"])

    # 3. Load Persistent FAISS Vector Index
    index_manager = FAISSIndexManager()
    index_manager.load_indices(load_flat=True, load_hnsw=True)
    if index_manager.hnsw_index is None and index_manager.flat_index is None:
        raise RuntimeError("No FAISS indices found on disk. Run indexing pipeline first.")

    app.state.index_manager = index_manager
    print("[Startup] FAISS indices and metadata lookup table loaded into memory.")

    yield

    # Teardown logic
    print("[Shutdown] Cleaning up visual search server resources.")


# Initialize FastAPI application
app = FastAPI(
    title="Production Metric Learning Visual Search Engine",
    description="Low-latency visual similarity search powered by ResNet-50 metric learning and FAISS HNSW.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mounts
raw_images_dir = PROJECT_ROOT / "data" / "raw"
if raw_images_dir.is_dir():
    app.mount("/static/images", StaticFiles(directory=str(raw_images_dir)), name="raw_images")

static_ui_dir = PROJECT_ROOT / "src" / "api" / "static"
if static_ui_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_ui_dir)), name="static_ui")


@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the single-page Visual Search Web Interface."""
    index_html = PROJECT_ROOT / "src" / "api" / "static" / "index.html"
    if index_html.is_file():
        return FileResponse(index_html)
    return {"message": "Visual search UI template not found."}


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check():
    """Health check endpoint reporting hardware and model status."""
    manager: FAISSIndexManager = app.state.index_manager
    total_vectors = manager.hnsw_index.ntotal if manager.hnsw_index else (
        manager.flat_index.ntotal if manager.flat_index else 0
    )
    return HealthResponse(
        status="healthy",
        model_loaded=(app.state.model is not None),
        index_loaded=(manager.hnsw_index is not None or manager.flat_index is not None),
        total_indexed_vectors=total_vectors,
        device=str(app.state.device),
    )


@app.get("/index-info", response_model=IndexInfoResponse, tags=["Diagnostics"])
async def index_info():
    """Retrieve details on the active FAISS vector index."""
    manager: FAISSIndexManager = app.state.index_manager
    cfg = app.state.cfg
    use_hnsw = manager.hnsw_index is not None
    active_index = manager.hnsw_index if use_hnsw else manager.flat_index

    return IndexInfoResponse(
        index_type="IndexHNSWFlat" if use_hnsw else "IndexFlatIP",
        dimension=cfg["model"]["embedding_dim"],
        total_vectors=active_index.ntotal if active_index else 0,
        metric="Inner Product (Cosine Similarity)",
        hnsw_m=cfg["indexing"].get("hnsw_m", 32) if use_hnsw else 0,
        hnsw_ef_search=cfg["indexing"].get("hnsw_ef_search", 64) if use_hnsw else 0,
    )


@app.post("/search", response_model=VisualSearchResponse, tags=["Search"])
async def search_similar_images(
    file: UploadFile = File(..., description="Uploaded query image file (JPEG, PNG, WebP)"),
    top_k: int = Query(5, ge=1, le=50, description="Number of top visual matches to retrieve"),
    use_approximate: bool = Query(True, description="Whether to use HNSW (True) or exact FlatIP (False)"),
):
    """
    Visual Search Query Endpoint.
    Accepts an uploaded image, generates a 512-D normalized embedding, queries FAISS,
    and returns top matching products with similarity scores and latency breakdown.
    """
    t_start = time.perf_counter()

    # 1. Format Validation
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported media type: '{file.content_type}'. Must be an image file.",
        )

    # 2. Image Decoding & Preprocessing
    t_pre_start = time.perf_counter()
    try:
        file_bytes = await file.read()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or invalid image file: {str(e)}",
        )

    transform = app.state.transform
    img_tensor = transform(image).unsqueeze(0).to(app.state.device)  # [1, 3, 224, 224]
    preprocess_ms = (time.perf_counter() - t_pre_start) * 1000

    # 3. Model Inference (Embedding Extraction)
    t_inf_start = time.perf_counter()
    with torch.inference_mode():
        # Output is strictly L2-normalized onto unit hypersphere
        query_embedding = app.state.model(img_tensor).cpu().numpy()  # [1, 512]
    inference_ms = (time.perf_counter() - t_inf_start) * 1000

    # 4. FAISS Vector Retrieval
    t_search_start = time.perf_counter()
    manager: FAISSIndexManager = app.state.index_manager
    raw_results = manager.search(
        query_embedding=query_embedding,
        top_k=top_k,
        use_approximate=use_approximate,
    )
    search_ms = (time.perf_counter() - t_search_start) * 1000
    total_ms = (time.perf_counter() - t_start) * 1000

    # Format result items with direct image URLs
    formatted_results = []
    for item in raw_results:
        rel_path = item.get("relative_path", "")
        formatted_results.append(
            SearchResultItem(
                rank=item["rank"],
                similarity_score=item["similarity_score"],
                product_id=item["product_id"],
                image_id=item["image_id"],
                relative_path=rel_path,
                image_url=f"/static/images/{rel_path}",
            )
        )

    return VisualSearchResponse(
        status="success",
        query_filename=file.filename or "uploaded_query.jpg",
        top_k=len(formatted_results),
        latency=LatencyBreakdown(
            preprocess_ms=round(preprocess_ms, 2),
            inference_ms=round(inference_ms, 2),
            search_ms=round(search_ms, 3),
            total_ms=round(total_ms, 2),
        ),
        results=formatted_results,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)

