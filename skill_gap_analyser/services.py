"""
Service Layer: Agentic Logic for PDF Processing and AI Extraction
"""
import logging
import fitz  # PyMuPDF
import asyncio
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from models import ExtractedSkillProfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFProcessingService:
    """Service for extracting text from PDF files"""
    
    @staticmethod
    async def extract_text_from_pdf(pdf_bytes: bytes) -> str:
        """
        Extract text from PDF using PyMuPDF
        
        Args:
            pdf_bytes: Raw PDF file bytes
            
        Returns:
            Extracted text content
            
        Raises:
            Exception: If PDF processing fails
        """
        try:
            logger.info("Starting PDF text extraction")
            
            # Run in thread pool to avoid blocking
            def extract():
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                text_content = []
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text_content.append(page.get_text())
                
                extracted_text = "\n".join(text_content)
                doc.close()
                return extracted_text
            
            extracted_text = await asyncio.to_thread(extract)
            
            logger.info(f"Successfully extracted {len(extracted_text)} characters from PDF")
            return extracted_text
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")


class AIExtractionService:
    """Service for AI-powered structured data extraction using Groq"""
    
    def __init__(self, groq_api_key: str):
        """
        Initialize AI extraction service
        
        Args:
            groq_api_key: Groq API key for authentication
        """
        self.groq_api_key = groq_api_key
        self.model_name = "llama-3.3-70b-versatile"
        
        # Initialize LLM with JSON mode
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=self.model_name,
            temperature=0.1,
            max_tokens=4096,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        # Define extraction prompt with JSON schema
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Healthcare IT recruiter and skills analyst. 
Your task is to extract structured information from resumes with a focus on healthcare interoperability.

Pay special attention to:
- FHIR (Fast Healthcare Interoperability Resources)
- HL7 (Health Level Seven) standards
- HIPAA (Health Insurance Portability and Accountability Act) compliance
- Healthcare integration projects
- Medical data systems
- EHR/EMR systems

Extract all technical skills, projects, and healthcare domain knowledge accurately.
For each project, identify the medical/healthcare impact clearly.

You MUST respond with valid JSON following this exact schema:
{{
  "full_name": "string",
  "technical_skills": ["skill1", "skill2"],
  "healthcare_domain_knowledge": {{
    "has_fhir": true/false,
    "has_hl7": true/false,
    "has_hipaa": true/false,
    "other_healthcare_standards": ["standard1", "standard2"]
  }},
  "projects": [
    {{
      "title": "string",
      "description": "string",
      "tech_stack": ["tech1", "tech2"],
      "medical_impact": "string"
    }}
  ],
  "years_of_experience": number or null,
  "certifications": ["cert1", "cert2"]
}}

Respond ONLY with valid JSON, no other text."""),
            ("human", "Extract structured information from this resume:\n\n{resume_text}")
        ])
        
        self.chain = self.prompt | self.llm
        logger.info("AI Extraction Service initialized successfully")
    
    async def extract_skills_from_text(self, resume_text: str) -> ExtractedSkillProfile:
        """
        Extract structured skill profile from resume text using AI
        
        Args:
            resume_text: Extracted text from resume
            
        Returns:
            ExtractedSkillProfile with structured data
            
        Raises:
            Exception: If AI extraction fails
        """
        try:
            logger.info("Starting AI-powered skill extraction")
            
            # Invoke the chain in thread pool
            result = await asyncio.to_thread(
                self.chain.invoke,
                {"resume_text": resume_text}
            )
            
            # Parse JSON response
            json_content = result.content
            
            # Clean up response if needed
            if "```json" in json_content:
                json_content = json_content.split("```json")[1].split("```")[0].strip()
            elif "```" in json_content:
                json_content = json_content.split("```")[1].split("```")[0].strip()
            
            parsed_data = json.loads(json_content)
            
            # Convert to Pydantic model
            profile = ExtractedSkillProfile(**parsed_data)
            
            logger.info(f"Successfully extracted profile for: {profile.full_name}")
            logger.info(f"Found {len(profile.technical_skills)} technical skills")
            logger.info(f"Found {len(profile.projects)} projects")
            
            return profile
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            logger.error(f"Raw response: {result.content if 'result' in locals() else 'No response'}")
            raise Exception(f"Failed to parse AI response as JSON: {str(e)}")
        except Exception as e:
            logger.error(f"AI extraction failed: {str(e)}")
            raise Exception(f"Failed to extract skills using AI: {str(e)}")