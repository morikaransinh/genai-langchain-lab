"""
Career Gap Analysis Service with Real Resource Search using SerpAPI
"""
import logging
import asyncio
import json
from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from serpapi import GoogleSearch

from career_gap_models import (
    CareerGapAnalysisResponse,
    SkillWithResources,
    LearningResource
)

logger = logging.getLogger(__name__)


class RealResourceSearchService:
    """Service for searching real learning resources using SerpAPI"""
    
    def __init__(self, serpapi_key: str):
        """
        Initialize Real Resource Search service
        
        Args:
            serpapi_key: SerpAPI key for authentication
        """
        self.serpapi_key = serpapi_key
        logger.info("Real Resource Search Service initialized")
    
    def search_youtube_videos(self, skill: str, max_results: int = 3) -> List[LearningResource]:
        """
        Search for real YouTube tutorial videos
        
        Args:
            skill: Skill to search for
            max_results: Maximum number of results
            
        Returns:
            List of YouTube video resources
        """
        try:
            search_query = f"{skill} tutorial healthcare"
            
            params = {
                "engine": "youtube",
                "search_query": search_query,
                "api_key": self.serpapi_key
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            resources = []
            video_results = results.get("video_results", [])[:max_results]
            
            for video in video_results:
                resource = LearningResource(
                    type="youtube",
                    title=video.get("title", ""),
                    url=video.get("link", ""),
                    author=video.get("channel", {}).get("name", "Unknown"),
                    description=video.get("description", f"YouTube tutorial on {skill}")[:200],
                    difficulty="beginner"
                )
                resources.append(resource)
            
            logger.info(f"Found {len(resources)} YouTube videos for {skill}")
            return resources
            
        except Exception as e:
            logger.error(f"YouTube search failed for {skill}: {str(e)}")
            return []
    
    def search_online_courses(self, skill: str, max_results: int = 2) -> List[LearningResource]:
        """
        Search for real online courses
        
        Args:
            skill: Skill to search for
            max_results: Maximum number of results
            
        Returns:
            List of online course resources
        """
        try:
            search_query = f"{skill} course udemy coursera healthcare"
            
            params = {
                "engine": "google",
                "q": search_query,
                "api_key": self.serpapi_key,
                "num": max_results * 2  # Get more results to filter
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            resources = []
            organic_results = results.get("organic_results", [])
            
            # Filter for course platforms
            course_platforms = ["udemy.com", "coursera.org", "edx.org", "linkedin.com/learning", "pluralsight.com"]
            
            for result in organic_results:
                link = result.get("link", "")
                if any(platform in link for platform in course_platforms):
                    resource = LearningResource(
                        type="course",
                        title=result.get("title", ""),
                        url=link,
                        author=result.get("displayed_link", "").split("/")[0],
                        description=result.get("snippet", f"Online course on {skill}")[:200],
                        difficulty="intermediate"
                    )
                    resources.append(resource)
                    
                    if len(resources) >= max_results:
                        break
            
            logger.info(f"Found {len(resources)} online courses for {skill}")
            return resources
            
        except Exception as e:
            logger.error(f"Course search failed for {skill}: {str(e)}")
            return []
    
    def search_books(self, skill: str, max_results: int = 2) -> List[LearningResource]:
        """
        Search for real books
        
        Args:
            skill: Skill to search for
            max_results: Maximum number of results
            
        Returns:
            List of book resources
        """
        try:
            search_query = f"{skill} book healthcare amazon"
            
            params = {
                "engine": "google",
                "q": search_query,
                "api_key": self.serpapi_key,
                "num": max_results * 2
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            resources = []
            organic_results = results.get("organic_results", [])
            
            # Filter for book platforms
            book_platforms = ["amazon.com", "books.google.com", "oreilly.com"]
            
            for result in organic_results:
                link = result.get("link", "")
                if any(platform in link for platform in book_platforms):
                    resource = LearningResource(
                        type="book",
                        title=result.get("title", ""),
                        url=link,
                        author="Various Authors",
                        description=result.get("snippet", f"Book on {skill}")[:200],
                        difficulty="intermediate"
                    )
                    resources.append(resource)
                    
                    if len(resources) >= max_results:
                        break
            
            logger.info(f"Found {len(resources)} books for {skill}")
            return resources
            
        except Exception as e:
            logger.error(f"Book search failed for {skill}: {str(e)}")
            return []
    
    async def get_resources_for_skill(self, skill: str) -> List[LearningResource]:
        """
        Get all types of resources for a skill
        
        Args:
            skill: Skill to search for
            
        Returns:
            Combined list of all resource types
        """
        try:
            # Run searches in parallel using asyncio
            youtube_task = asyncio.to_thread(self.search_youtube_videos, skill, 3)
            courses_task = asyncio.to_thread(self.search_online_courses, skill, 2)
            books_task = asyncio.to_thread(self.search_books, skill, 2)
            
            youtube_results, course_results, book_results = await asyncio.gather(
                youtube_task, courses_task, books_task
            )
            
            # Combine all resources
            all_resources = youtube_results + course_results + book_results
            
            logger.info(f"Total resources found for {skill}: {len(all_resources)}")
            return all_resources
            
        except Exception as e:
            logger.error(f"Failed to get resources for {skill}: {str(e)}")
            return []


class CareerGapAnalysisService:
    """Service for analyzing career gaps and providing learning resources"""
    
    def __init__(self, groq_api_key: str, serpapi_key: str):
        """
        Initialize Career Gap Analysis service
        
        Args:
            groq_api_key: Groq API key for authentication
            serpapi_key: SerpAPI key for real resource search
        """
        self.groq_api_key = groq_api_key
        self.serpapi_key = serpapi_key
        self.model_name = "llama-3.3-70b-versatile"
        
        # Initialize LLM with JSON mode
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=self.model_name,
            temperature=0.2,
            max_tokens=4096,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        # Initialize Real Resource Search
        self.resource_search = RealResourceSearchService(serpapi_key)
        
        # Prompt for skill gap analysis (simplified - no resource generation)
        self.gap_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Healthcare IT career advisor specializing in skill gap analysis.

Your task is to:
1. Analyze the user's current skills
2. Identify required skills for their target healthcare role
3. Calculate the skill gap
4. Classify each missing skill's importance (critical, important, nice-to-have)

Focus on healthcare-specific skills like:
- FHIR, HL7, HIPAA, ICD-10, DICOM
- Healthcare systems: EHR/EMR, PACS, HIS
- Healthcare data standards and interoperability
- Medical terminology and clinical workflows

You MUST respond with valid JSON following this exact schema:
{{
  "required_skills": ["skill1", "skill2", "skill3"],
  "skill_gap": [
    {{
      "skill_name": "FHIR",
      "importance": "critical"
    }},
    {{
      "skill_name": "HL7",
      "importance": "important"
    }}
  ],
  "recommendations": "Based on your background, you should first focus on..."
}}

Respond ONLY with valid JSON. Do not include learning resources - those will be searched separately."""),
            ("human", """
Current Skills: {current_skills}
Target Career Goal: {career_goal}

Please analyze the skill gap.
""")
        ])
        
        self.chain = self.gap_analysis_prompt | self.llm
        logger.info("Career Gap Analysis Service initialized successfully")
    
    async def analyze_career_gap(
        self,
        current_skills: List[str],
        career_goal: str
    ) -> dict:
        """
        Analyze career gap
        
        Args:
            current_skills: List of all current skills
            career_goal: Target healthcare role
            
        Returns:
            Dictionary with gap analysis
        """
        try:
            logger.info(f"Analyzing career gap for goal: {career_goal}")
            logger.info(f"Current skills count: {len(current_skills)}")
            
            # Invoke AI to analyze gap
            result = await asyncio.to_thread(
                self.chain.invoke,
                {
                    "current_skills": ", ".join(current_skills),
                    "career_goal": career_goal
                }
            )
            
            # Parse JSON response
            json_content = result.content
            
            # Clean up response
            if "```json" in json_content:
                json_content = json_content.split("```json")[1].split("```")[0].strip()
            elif "```" in json_content:
                json_content = json_content.split("```")[1].split("```")[0].strip()
            
            parsed_data = json.loads(json_content)
            
            logger.info(f"Gap analysis completed: {len(parsed_data.get('skill_gap', []))} skills identified")
            
            return parsed_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            raise Exception(f"Failed to parse AI response: {str(e)}")
        except Exception as e:
            logger.error(f"Career gap analysis failed: {str(e)}")
            raise Exception(f"Failed to analyze career gap: {str(e)}")
    
    async def process_gap_analysis(
        self,
        resume_data: dict,
        additional_skills: List[str],
        career_goal: str
    ) -> CareerGapAnalysisResponse:
        """
        Complete gap analysis workflow with real resource search
        
        Args:
            resume_data: Resume data from /predict endpoint
            additional_skills: User-provided additional skills
            career_goal: Target healthcare role
            
        Returns:
            Complete gap analysis response with real resources
        """
        try:
            # Combine all current skills
            technical_skills = resume_data.get("technical_skills", [])
            
            # Add healthcare domain knowledge
            healthcare_knowledge = resume_data.get("healthcare_domain_knowledge", {})
            if healthcare_knowledge.get("has_fhir"):
                technical_skills.append("FHIR")
            if healthcare_knowledge.get("has_hl7"):
                technical_skills.append("HL7")
            if healthcare_knowledge.get("has_hipaa"):
                technical_skills.append("HIPAA")
            
            # Add other healthcare standards
            other_standards = healthcare_knowledge.get("other_healthcare_standards", [])
            technical_skills.extend(other_standards)
            
            # Add certifications as skills
            certifications = resume_data.get("certifications", [])
            technical_skills.extend(certifications)
            
            # Combine with additional skills
            current_skills = list(set(technical_skills + additional_skills))
            
            logger.info(f"Total current skills: {len(current_skills)}")
            
            # Step 1: Analyze gap using AI (no resources yet)
            gap_data = await self.analyze_career_gap(current_skills, career_goal)
            
            # Parse required skills
            required_skills = gap_data.get("required_skills", [])
            
            # Parse skill gaps
            skill_gap_data = gap_data.get("skill_gap", [])
            
            # Step 2: Search for REAL resources for each missing skill using SerpAPI
            logger.info(f"Searching for real resources for {len(skill_gap_data)} skills...")
            
            skill_gaps = []
            for gap in skill_gap_data:
                skill_name = gap.get("skill_name")
                importance = gap.get("importance")
                
                # Search for real resources using SerpAPI
                resources = await self.resource_search.get_resources_for_skill(skill_name)
                
                skill_gaps.append(SkillWithResources(
                    skill_name=skill_name,
                    importance=importance,
                    resources=resources
                ))
            
            # Calculate matching skills
            required_set = set([s.lower() for s in required_skills])
            current_set = set([s.lower() for s in current_skills])
            matching_skills = list(required_set.intersection(current_set))
            
            # Calculate gap percentage
            if required_skills:
                gap_percentage = (len(skill_gaps) / len(required_skills)) * 100
            else:
                gap_percentage = 0.0
            
            logger.info(f"✅ Gap analysis complete with real resources")
            
            return CareerGapAnalysisResponse(
                success=True,
                message="Career gap analysis completed successfully with real learning resources",
                career_goal=career_goal,
                current_skills=current_skills,
                required_skills=required_skills,
                skill_gap=skill_gaps,
                matching_skills=matching_skills,
                gap_percentage=round(gap_percentage, 2),
                recommendations=gap_data.get("recommendations", "")
            )
            
        except Exception as e:
            logger.error(f"Gap analysis processing failed: {str(e)}")
            raise Exception(f"Failed to process gap analysis: {str(e)}")