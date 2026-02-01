"""
FastAPI Application: Healthcare Career Intelligence System
Complete system with Resume Analysis + Career Gap Analysis
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Body
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from models import PredictResponse
from services import PDFProcessingService, AIExtractionService
from career_gap_models import CareerGapAnalysisRequest, CareerGapAnalysisResponse
from career_gap_service import CareerGapAnalysisService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instances
pdf_service = PDFProcessingService()
ai_service = None
career_gap_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    global ai_service, career_gap_service
    
    # Startup: Initialize services
    logger.info("=" * 60)
    logger.info("Initializing Healthcare Career Intelligence System...")
    logger.info("=" * 60)
    
    try:
        # Validate environment variables
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        logger.info(f"✓ Environment variables loaded")
        
        # Initialize AI service
        ai_service = AIExtractionService(groq_api_key=groq_api_key)
        logger.info(f"✓ AI Service initialized (Model: llama-3.3-70b-versatile)")
        
        # Get SerpAPI key
        serpapi_key = os.getenv("SERPAPI_KEY")
        if not serpapi_key:
            logger.warning("⚠️ SERPAPI_KEY not found - resource search will be limited")
            serpapi_key = None
        
        # Initialize Career Gap Analysis service
        career_gap_service = CareerGapAnalysisService(
            groq_api_key=groq_api_key,
            serpapi_key=serpapi_key
        )
        logger.info(f"✓ Career Gap Analysis Service initialized")
        
        logger.info("=" * 60)
        logger.info("✅ All services initialized successfully!")
        logger.info("=" * 60)
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {str(e)}")
        raise
    
    finally:
        logger.info("Shutting down services...")
        logger.info("Shutdown complete")


# Initialize FastAPI application
app = FastAPI(
    title="Healthcare Career Intelligence System",
    description="AI-powered resume analysis and career gap analysis for healthcare professionals",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Healthcare Career Intelligence System",
        "version": "2.0.0",
        "endpoints": {
            "predict": "/predict - Extract skills from resume PDF",
            "analyze_gap": "/analyze-career-gap - Analyze career gap and get learning resources",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict(file: UploadFile = File(...)):
    """
    Upload and analyze a resume PDF
    
    Args:
        file: PDF file upload
        
    Returns:
        PredictResponse with extracted profile data
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are accepted"
            )
        
        logger.info(f"Processing resume upload: {file.filename}")
        
        # Read file bytes
        file_bytes = await file.read()
        
        if len(file_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )
        
        logger.info(f"File size: {len(file_bytes)} bytes")
        
        # Step 1: Extract text from PDF
        try:
            resume_text = await pdf_service.extract_text_from_pdf(file_bytes)
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to extract text from PDF: {str(e)}"
            )
        
        if not resume_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No text content found in PDF"
            )
        
        # Step 2: Extract structured skills using AI
        try:
            extracted_profile = await ai_service.extract_skills_from_text(resume_text)
        except Exception as e:
            logger.error(f"AI extraction failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to extract skills: {str(e)}"
            )
        
        logger.info(f"✅ Resume analysis completed successfully for: {extracted_profile.full_name}")
        
        return PredictResponse(
            success=True,
            message="Resume analyzed successfully",
            data=extracted_profile
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in resume analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/analyze-career-gap", response_model=CareerGapAnalysisResponse, status_code=status.HTTP_200_OK)
async def analyze_career_gap(request: CareerGapAnalysisRequest = Body(...)):
    """
    Analyze career gap based on resume data and career goal
    
    Args:
        request: CareerGapAnalysisRequest with resume_data, additional_skills, and career_goal
        
    Returns:
        CareerGapAnalysisResponse with gap analysis and learning resources
        
    Example Request:
    {
        "resume_data": {
            "full_name": "John Doe",
            "technical_skills": ["Python", "Machine Learning"],
            "healthcare_domain_knowledge": {
                "has_fhir": false,
                "has_hl7": false,
                "has_hipaa": false,
                "other_healthcare_standards": []
            },
            "projects": [],
            "years_of_experience": 2,
            "certifications": []
        },
        "additional_skills": ["Docker", "Kubernetes"],
        "career_goal": "FHIR Integration Specialist"
    }
    """
    try:
        logger.info(f"Processing career gap analysis for goal: {request.career_goal}")
        
        if not request.career_goal.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Career goal cannot be empty"
            )
        
        # Process gap analysis
        try:
            gap_analysis = await career_gap_service.process_gap_analysis(
                resume_data=request.resume_data,
                additional_skills=request.additional_skills,
                career_goal=request.career_goal
            )
        except Exception as e:
            logger.error(f"Gap analysis failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to analyze career gap: {str(e)}"
            )
        
        logger.info(f"✅ Career gap analysis completed: {gap_analysis.gap_percentage}% gap identified")
        
        return gap_analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in career gap analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """
    Detailed health check endpoint
    """
    from datetime import datetime
    
    health_status = {
        "status": "healthy",
        "services": {
            "pdf_processing": "operational",
            "ai_extraction": "operational" if ai_service else "unavailable",
            "career_gap_analysis": "operational" if career_gap_service else "unavailable"
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return health_status


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", 8000))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )