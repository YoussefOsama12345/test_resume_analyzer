import re
import json
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from rapidfuzz import process, fuzz
from dataclasses import dataclass
from datetime import datetime
from ai.extractors.resume.experiance_extractor import ResumeExperienceExtractor
from ai.extractors.job.experiance_extractor import JobExperienceExtractor

@dataclass
class ExperienceMatch:
    title: str
    normalized_title: Optional[str]
    years: int
    required: bool
    preferred: bool
    confidence: float
    field: Optional[str]
    skills: List[str]
    level: Optional[str]

@dataclass
class ExperienceComparison:
    job_title: str
    resume_title: str
    job_years: int
    resume_years: int
    title_match_score: float
    years_match: bool
    skills_match: List[str]
    missing_skills: List[str]
    level_match: bool
    overall_match_score: float

class ExperienceMatcher:
    def __init__(self, experience_file: str = "data/experience.json"):
        self.resume_extractor = ResumeExperienceExtractor(experience_file)
        self.job_extractor = JobExperienceExtractor(experience_file)
        self.experience_db = self._load_experience_data(experience_file)
        self.title_lookup = self._build_title_lookup()
        self.skill_patterns = [
            r"(?:proficiency|experience|expertise|knowledge|skills?)\s+(?:in|with|of)\s+([a-zA-Z\s,&/\\-]+)",
            r"(?:strong|solid|excellent|good|deep)\s+(?:knowledge|understanding|experience|proficiency)\s+(?:in|with|of)\s+([a-zA-Z\s,&/\\-]+)",
            r"(?:familiarity|familiar)\s+(?:with|in)\s+([a-zA-Z\s,&/\\-]+)",
        ]
        self.required_keywords = {
            "required": ["required", "must have", "mandatory", "essential", "minimum"],
            "preferred": ["preferred", "nice to have", "desired", "bonus", "plus"],
            "years": ["years", "yrs", "year", "yr"],
            "experience": ["experience", "exp", "expertise", "proficiency"]
        }

    def _load_experience_data(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_title_lookup(self) -> Dict[str, Dict]:
        lookup = {}
        for item in self.experience_db:
            key = item["title"].lower().strip()
            lookup[key] = item
            for alias in item.get("aliases", []):
                lookup[alias.lower().strip()] = item
        return lookup

    def _extract_skills(self, text: str) -> List[str]:
        skills = set()
        for pattern in self.skill_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                skill = match.group(1).strip().lower()
                if skill and len(skill.split()) <= 5:  # Avoid long phrases
                    skills.add(skill)
        return list(skills)

    def _extract_years(self, text: str) -> Optional[int]:
        # Match various year formats
        patterns = [
            r"(\d{1,2})\+?\s*(?:years?|yrs?)",
            r"minimum\s+of\s+(\d{1,2})\+?\s*(?:years?|yrs?)",
            r"at least\s+(\d{1,2})\+?\s*(?:years?|yrs?)",
            r"(\d{1,2})\+?\s*(?:years?|yrs?)\s+of\s+experience"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def _is_required(self, text: str) -> bool:
        return any(kw in text.lower() for kw in self.required_keywords["required"])

    def _is_preferred(self, text: str) -> bool:
        return any(kw in text.lower() for kw in self.required_keywords["preferred"])

    def _calculate_confidence(self, text: str, years: Optional[int]) -> float:
        confidence = 0.5  # Base confidence
        
        # Adjust based on presence of key indicators
        if years is not None:
            confidence += 0.2
        if any(kw in text.lower() for kw in self.required_keywords["experience"]):
            confidence += 0.1
        if any(kw in text.lower() for kw in self.required_keywords["years"]):
            confidence += 0.1
        if self._is_required(text) or self._is_preferred(text):
            confidence += 0.1
            
        return min(confidence, 1.0)

    def match(self, text: str) -> Optional[ExperienceMatch]:
        # Extract years
        years = self._extract_years(text)
        if not years:
            return None

        # Extract skills
        skills = self._extract_skills(text)

        # Try to match title
        title = None
        normalized_title = None
        field = None
        level = None

        # Look for title patterns in the text
        title_patterns = [
            r"(?:experience|expertise|proficiency)\s+(?:in|with|as)\s+([a-zA-Z\s,&/\\-]+)",
            r"(?:role|position|job)\s+(?:of|as)\s+([a-zA-Z\s,&/\\-]+)",
            r"([a-zA-Z\s,&/\\-]+(?:engineer|scientist|developer|analyst|architect|manager|consultant|specialist))"
        ]

        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                # Try to normalize the title
                matched = self._match_title(title)
                if matched:
                    normalized_title = matched["title"]
                    field = matched.get("field")
                    level = self._extract_level(title)
                break

        if not title:
            return None

        return ExperienceMatch(
            title=title,
            normalized_title=normalized_title,
            years=years,
            required=self._is_required(text),
            preferred=self._is_preferred(text),
            confidence=self._calculate_confidence(text, years),
            field=field,
            skills=skills,
            level=level
        )

    def _match_title(self, text: str) -> Optional[Dict]:
        text = text.lower().strip()
        
        # Try exact match first
        if text in self.title_lookup:
            return self.title_lookup[text]
            
        # Try fuzzy matching
        best_match, score, _ = process.extractOne(
            text,
            self.title_lookup.keys(),
            scorer=fuzz.token_sort_ratio
        )
        if score >= 85:
            return self.title_lookup[best_match]
            
        return None

    def _extract_level(self, title: str) -> Optional[str]:
        title_lower = title.lower()
        if "senior" in title_lower or "sr" in title_lower:
            return "senior"
        elif "junior" in title_lower or "jr" in title_lower:
            return "junior"
        elif "lead" in title_lower:
            return "lead"
        elif "principal" in title_lower:
            return "principal"
        elif "staff" in title_lower:
            return "staff"
        return None

    def _calculate_title_match_score(self, job_title: str, resume_title: str) -> float:
        # Try exact match first
        if job_title.lower() == resume_title.lower():
            return 1.0
            
        # Try fuzzy matching with different strategies
        strategies = [
            (fuzz.token_sort_ratio, 0.7),
            (fuzz.token_set_ratio, 0.8),
            (fuzz.partial_ratio, 0.9)
        ]
        
        max_score = 0.0
        for scorer, weight in strategies:
            score = scorer(job_title.lower(), resume_title.lower()) / 100.0
            max_score = max(max_score, score * weight)
            
        return max_score

    def _calculate_skills_match(self, job_skills: List[str], resume_skills: List[str]) -> Tuple[List[str], List[str]]:
        job_skills_set = {skill.lower() for skill in job_skills}
        resume_skills_set = {skill.lower() for skill in resume_skills}
        
        matching_skills = list(job_skills_set.intersection(resume_skills_set))
        missing_skills = list(job_skills_set - resume_skills_set)
        
        return matching_skills, missing_skills

    def _calculate_years_match(self, job_years: int, resume_years: int) -> bool:
        return resume_years >= job_years

    def _calculate_level_match(self, job_level: Optional[str], resume_level: Optional[str]) -> bool:
        if not job_level or not resume_level:
            return True  # If level is not specified, consider it a match
            
        level_hierarchy = {
            "junior": 1,
            "staff": 2,
            "senior": 3,
            "lead": 4,
            "principal": 5
        }
        
        job_level_value = level_hierarchy.get(job_level.lower(), 0)
        resume_level_value = level_hierarchy.get(resume_level.lower(), 0)
        
        return resume_level_value >= job_level_value

    def _calculate_overall_match_score(self, title_score: float, years_match: bool, 
                                     skills_match: List[str], total_skills: int,
                                     level_match: bool) -> float:
        weights = {
            "title": 0.4,
            "years": 0.3,
            "skills": 0.2,
            "level": 0.1
        }
        
        score = 0.0
        score += title_score * weights["title"]
        score += (1.0 if years_match else 0.0) * weights["years"]
        score += (len(skills_match) / max(total_skills, 1)) * weights["skills"]
        score += (1.0 if level_match else 0.0) * weights["level"]
        
        return score

    def match_experiences(self, job_text: str, resume_text: str) -> List[ExperienceComparison]:
        # Extract job requirements
        job_data = self.job_extractor.extract(job_text)
        job_requirements = job_data["experience_requirements"]
        
        # Extract resume experiences
        resume_experiences = self.resume_extractor.extract(resume_text)
        
        comparisons = []
        
        for job_req in job_requirements:
            best_match = None
            best_score = 0.0
            
            for resume_exp in resume_experiences:
                # Calculate title match score
                title_score = self._calculate_title_match_score(
                    job_req["title"],
                    resume_exp["job_title"]
                )
                
                # Calculate years match using duration_years from resume experience
                resume_years = resume_exp.get("duration_years", 0)
                years_match = self._calculate_years_match(job_req["min_years"], resume_years)
                
                # Calculate skills match
                matching_skills, missing_skills = self._calculate_skills_match(
                    job_req.get("skills", []),
                    resume_exp.get("skills", [])
                )
                
                # Calculate level match
                level_match = self._calculate_level_match(
                    job_req.get("level"),
                    resume_exp.get("level")
                )
                
                # Calculate overall match score
                overall_score = self._calculate_overall_match_score(
                    title_score,
                    years_match,
                    matching_skills,
                    len(job_req.get("skills", [])),
                    level_match
                )
                
                if overall_score > best_score:
                    best_score = overall_score
                    best_match = ExperienceComparison(
                        job_title=job_req["title"],
                        resume_title=resume_exp["job_title"],
                        job_years=job_req["min_years"],
                        resume_years=resume_years,
                        title_match_score=title_score,
                        years_match=years_match,
                        skills_match=matching_skills,
                        missing_skills=missing_skills,
                        level_match=level_match,
                        overall_match_score=overall_score
                    )
            
            if best_match:
                comparisons.append(best_match)
        
        return comparisons

def main():
    # Set your input file paths
    job_path = Path("inputs/jobs/job1.txt")
    resume_path = Path("inputs/resumes/resume1.txt")
    experience_db_path = "data/experience.json"

    # Check if files exist
    if not job_path.exists():
        print(f"❌ Job description file not found: {job_path}")
        return
    if not resume_path.exists():
        print(f"❌ Resume file not found: {resume_path}")
        return

    # Load texts
    job_text = job_path.read_text(encoding="utf-8")
    resume_text = resume_path.read_text(encoding="utf-8")
    
    print(job_text)
    print("--------------------------------")
    print(resume_text)
    print("--------------------------------")

    # Initialize matcher
    matcher = ExperienceMatcher(experience_file=experience_db_path)

    # Match experiences
    matches = matcher.match_experiences(job_text, resume_text)

    # Display results
    if matches:
        print(f"\n✅ Found {len(matches)} experience matches:\n")
        for i, match in enumerate(matches, start=1):
            print(f"🔹 Match {i}:")
            print(f"   Job Title: {match.job_title}")
            print(f"   Resume Title: {match.resume_title}")
            print(f"   Title Match Score: {match.title_match_score:.2f}")
            print(f"   Years Match: {'✅' if match.years_match else '❌'}")
            print(f"   Job Years Required: {match.job_years}")
            print(f"   Resume Years: {match.resume_years}")
            print(f"   Matching Skills: {', '.join(match.skills_match) if match.skills_match else 'None'}")
            print(f"   Missing Skills: {', '.join(match.missing_skills) if match.missing_skills else 'None'}")
            print(f"   Level Match: {'✅' if match.level_match else '❌'}")
            print(f"   Overall Match Score: {match.overall_match_score:.2f}\n")
    else:
        print("⚠️ No experience matches found.")

if __name__ == "__main__":
    main()
