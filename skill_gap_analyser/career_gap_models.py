"""
Pydantic Models for Career Gap Analysis System
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class LearningResource(BaseModel):
    """Learning resource for a specific skill"""
    type: str = Field(..., description="Type: book, youtube, course")
    title: str = Field(..., description="Resource title")
    url: Optional[str] = Field(None, description="URL to resource")
    author: Optional[str] = Field(None, description="Author or creator")
    description: str = Field(..., description="Brief description")
    difficulty: str = Field(..., description="beginner, intermediate, advanced")


class SkillWithResources(BaseModel):
    """A missing skill with learning resources"""
    skill_name: str = Field(..., description="Name of the missing skill")
    importance: str = Field(..., description="critical, important, nice-to-have")
    resources: List[LearningResource] = Field(default_factory=list, description="Learning resources")


class CareerGapAnalysisRequest(BaseModel):
    """Request for career gap analysis"""
    resume_data: dict = Field(..., description="Data from /predict endpoint")
    additional_skills: List[str] = Field(default_factory=list, description="User-provided additional skills")
    career_goal: str = Field(..., description="Target healthcare role/position")


class CareerGapAnalysisResponse(BaseModel):
    """Response with gap analysis and learning resources"""
    success: bool
    message: str
    career_goal: str
    current_skills: List[str] = Field(default_factory=list, description="All current skills combined")
    required_skills: List[str] = Field(default_factory=list, description="Skills needed for target role")
    skill_gap: List[SkillWithResources] = Field(default_factory=list, description="Missing skills with resources")
    matching_skills: List[str] = Field(default_factory=list, description="Skills user already has")
    gap_percentage: float = Field(..., description="Percentage of skills missing")
    recommendations: str = Field(..., description="Career advice and next steps")