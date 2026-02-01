"""
Pydantic Models for Healthcare Skill Intelligence System
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class ProjectInfo(BaseModel):
    """Individual project information extracted from resume"""
    title: str = Field(..., description="Project title")
    description: str = Field(..., description="Detailed project description")
    tech_stack: List[str] = Field(default_factory=list, description="Technologies used")
    medical_impact: str = Field(..., description="Healthcare/medical impact of the project")


class HealthcareDomainKnowledge(BaseModel):
    """Healthcare-specific domain knowledge"""
    has_fhir: bool = Field(default=False, description="FHIR standard knowledge")
    has_hl7: bool = Field(default=False, description="HL7 standard knowledge")
    has_hipaa: bool = Field(default=False, description="HIPAA compliance knowledge")
    other_healthcare_standards: List[str] = Field(default_factory=list, description="Other healthcare standards")


class ExtractedSkillProfile(BaseModel):
    """
    Structured output schema for LLM extraction.
    """
    full_name: str = Field(..., description="Full name of the candidate")
    technical_skills: List[str] = Field(default_factory=list, description="List of technical skills")
    healthcare_domain_knowledge: HealthcareDomainKnowledge = Field(
        default_factory=HealthcareDomainKnowledge,
        description="Healthcare domain expertise"
    )
    projects: List[ProjectInfo] = Field(default_factory=list, description="List of projects")
    years_of_experience: Optional[int] = Field(None, description="Total years of experience")
    certifications: List[str] = Field(default_factory=list, description="Professional certifications")


class PredictResponse(BaseModel):
    """API response for resume prediction/analysis"""
    success: bool
    message: str
    data: Optional[ExtractedSkillProfile] = None