import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
from routers import evaluate


app = FastAPI(
    title="AI Resume Evaluator API",
    description=(
        "A resume-to-job-description matching API using structured extraction, "
        "semantic evidence retrieval, and evidence-based requirement verification. "
        "Scores represent weighted requirement coverage, not hiring probability."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://resumeeeval.vercel.app",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(evaluate.router, prefix="/api/v1", tags=["Evaluation"])


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "true").lower() == "true",
    )
