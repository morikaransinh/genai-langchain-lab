"""
FastAPI Application: Healthcare Career Intelligence System
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Body
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# ✅ FIXED IMPORTS
from skill_gap_analyser.models import PredictResponse
from skill_gap_analyser.services import PDFProcessingService, AIExtractionService
from skill_gap_analyser.career_gap_models import CareerGapAnalysisRequest, CareerGapAnalysisResponse
from skill_gap_analyser.career_gap_service import CareerGapAnalysisService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global services
pdf_service = PDFProcessingService()
ai_service = None
career_gap_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ai_service, career_gap_service

    logger.info("Initializing Healthcare Career Intelligence System...")

    try:
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found")

        # ✅ FIX: AIExtractionService imported
        ai_service = AIExtractionService(groq_api_key=groq_api_key)

        serpapi_key = os.getenv("SERPAPI_KEY")

        career_gap_service = CareerGapAnalysisService(
            groq_api_key=groq_api_key,
            serpapi_key=serpapi_key
        )

        logger.info("All services initialized successfully")

        yield

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


app = FastAPI(
    title="Healthcare Career Intelligence System",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(400, "Only PDF allowed")

        file_bytes = await file.read()

        resume_text = await pdf_service.extract_text_from_pdf(file_bytes)

        extracted_profile = await ai_service.extract_skills_from_text(resume_text)

        return PredictResponse(
            success=True,
            message="Success",
            data=extracted_profile
        )

    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/analyze-career-gap", response_model=CareerGapAnalysisResponse)
async def analyze_career_gap(request: CareerGapAnalysisRequest):
    try:
        result = await career_gap_service.process_gap_analysis(
            resume_data=request.resume_data,
            additional_skills=request.additional_skills,
            career_goal=request.career_goal
        )
        return result

    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
