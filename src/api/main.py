from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from src.analytics.visualization import generate_report
from src.rag.query_engine import BookingQA
from src.rag.vector_db import VectorDB
from src.api.schemas import AnalyticsRequest, QuestionRequest, AnalyticsResponse, HealthResponse
import logging
import base64
from fastapi.middleware.cors import CORSMiddleware  # Critical addition
from fastapi import Depends, Request, Query, HTTPException, Body,status
from fastapi.responses import JSONResponse
import json
from fastapi import Depends
# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api")

# Lifespan handler for resource management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    try:
        app.vector_db = VectorDB()
        app.qa_system = BookingQA()
        logging.info("System initialized successfully")
        yield
    except Exception as e:
        logging.error(f"Initialization failed: {str(e)}")
        raise
    # Shutdown logic (if needed)
    finally:
        logging.info("Shutting down...")
        # Add any cleanup logic here

app = FastAPI(
    title="Hotel Booking Analytics & QA",
    lifespan=lifespan
)
# Add CORS middleware - Critical for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#logger = logging.getLogger("api")

def encode_plot(fig):
    """Convert matplotlib figure to base64"""
    try:
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to encode plot: {str(e)}")
        return None


@app.get("/analytics")
async def analytics_endpoint(request: AnalyticsRequest):
    #logger.debug(f"Raw request: {await request.body()}")
    try:
        # Generate report with filters
        report = generate_report(
            metric=request.metric,
        )

        return {
            "status": "success",
            "visualization": report
        }
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        logger.error(f"Analytics error: {str(e)}")
        raise HTTPException(500, detail="Internal server error")


@app.api_route("/ask", methods=["GET", "POST"])
async def ask_endpoint(
        request: Request,
        question: str = Query(None),
        max_results: int = Query(3),
        body: QuestionRequest = None
):
    try:
        # Handle GET requests
        if request.method == "GET":
            if not question:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"message": "Add your question as a URL parameter: /ask?question=YOUR_QUESTION"}
                )
            query_text = question
            result_limit = max_results

        # Handle POST requests
        elif request.method == "POST":
            if not body:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid request body format"
                )
            query_text = body.question
            result_limit = body.max_results

        else:
            raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED)

        # Process the query
        response = app.qa_system.ask(
            question=query_text,
            max_results=result_limit
        )

        return {
            "question": query_text,
            "answer": response["answer"],
            "sources": [doc.metadata for doc in response["source_documents"]]
        }

    except HTTPException as he:
        # Re-raise HTTP exceptions (400/405/etc) directly
        raise he

    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error - check logs for details"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        return {
            "status": "healthy",
            "vector_db_records": app.vector_db.get_vector_store().index.ntotal,
            "llm_status": "ready"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/test")
async def test_endpoint():
    return {"message": "API is working!"}

"""if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",
        loop="asyncio",  # Force IPv4-compatible event loop"""
