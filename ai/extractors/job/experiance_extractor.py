import re
import json
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from rapidfuzz import process, fuzz
from dataclasses import dataclass
from datetime import datetime

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

class ExperienceMatcher:
    def __init__(self, experience_file: str = "data/experience.json"):
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

class JobExperienceExtractor:
    def __init__(self, experience_file: str = "data/experience.json"):
        self.experience_db = self._load_experience_data(experience_file)
        self.title_lookup = self._build_title_lookup()
        self.experience_matcher = ExperienceMatcher(experience_file)
        self.title_patterns = [
            # Primary job title patterns
            r"(?:job title|position|role|title)\s*:?\s*([a-zA-Z\s,&/\\-]+(?:engineer|scientist|developer|analyst|architect|manager|consultant|specialist))",
            r"(?:looking for|seeking|hiring)\s+(?:a|an)?\s+(?:highly skilled|experienced)?\s+(?:senior|junior|lead|principal|staff)?\s+([a-zA-Z\s,&/\\-]+(?:engineer|scientist|developer|analyst|architect|manager|consultant|specialist))",
            r"(?:we are|we're)\s+(?:looking for|seeking|hiring)\s+(?:a|an)?\s+(?:highly skilled|experienced)?\s+(?:senior|junior|lead|principal|staff)?\s+([a-zA-Z\s,&/\\-]+(?:engineer|scientist|developer|analyst|architect|manager|consultant|specialist))",
            # Role-specific patterns
            r"(?:senior|junior|lead|principal|staff)\s+([a-zA-Z\s,&/\\-]+(?:engineer|scientist|developer|analyst|architect|manager|consultant|specialist))",
            r"([a-zA-Z\s,&/\\-]+(?:engineer|scientist|developer|analyst|architect|manager|consultant|specialist))",
        ]
        self.experience_patterns = [
            r"(\d{1,2}\+?)\s*(?:years|yrs|year|yr)s?\s*(?:of)?\s*experience(?:\s+in|\s+with|\s+as)?\s+([a-zA-Z\s,&/\\-]+)",
            r"experience\s+of\s+(\d{1,2}\+?)\s*(?:years|yrs|year|yr)s?\s*(?:in|\s+with|\s+as)?\s+([a-zA-Z\s,&/\\-]+)",
            r"minimum\s+of\s+(\d{1,2}\+?)\s*(?:years|yrs|year|yr)s?\s*(?:of)?\s*experience(?:\s+in|\s+with|\s+as)?\s+([a-zA-Z\s,&/\\-]+)",
        ]
        self.ignored_patterns = [
            r"key responsibilities",
            r"about the role",
            r"job description",
            r"requirements",
            r"qualifications",
            r"responsibilities",
            r"duties",
            r"skills",
        ]

    def _load_experience_data(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_title_lookup(self) -> Dict[str, Dict]:
        lookup = {}
        for item in self.experience_db:
            # Add main title
            key = item["title"].lower().strip()
            lookup[key] = item
            
            # Add aliases
            for alias in item.get("aliases", []):
                lookup[alias.lower().strip()] = item
                
            # Add common variations
            if "senior" in key:
                lookup[key.replace("senior", "sr")] = item
            if "junior" in key:
                lookup[key.replace("junior", "jr")] = item
            if "lead" in key:
                lookup[key.replace("lead", "principal")] = item
                
        return lookup

    def _match_title(self, text: str) -> Optional[Dict]:
        text = text.lower().strip()
        
        # Try exact match first
        if text in self.title_lookup:
            return self.title_lookup[text]
            
        # Try fuzzy matching with different strategies
        strategies = [
            (fuzz.token_sort_ratio, 85),
            (fuzz.token_set_ratio, 90),
            (fuzz.partial_ratio, 95)
        ]
        
        for scorer, threshold in strategies:
            best_match, score, _ = process.extractOne(
                text,
                self.title_lookup.keys(),
                scorer=scorer
            )
            if score >= threshold:
                return self.title_lookup[best_match]
                
        return None

    def _is_valid_title(self, text: str) -> bool:
        # Check if the text contains any ignored patterns
        if any(re.search(pattern, text.lower()) for pattern in self.ignored_patterns):
            return False
            
        # Check if the text is too long (likely not a title)
        if len(text.split()) > 10:
            return False
            
        # Check if the text contains common job title indicators
        title_indicators = ["engineer", "scientist", "developer", "analyst", "architect", "manager", "consultant", "specialist"]
        return any(indicator in text.lower() for indicator in title_indicators)

    def _deduplicate_titles(self, titles: List[Dict]) -> List[Dict]:
        seen_titles: Set[str] = set()
        unique_titles = []
        
        for title in titles:
            # Normalize the title for comparison
            normalized = title["normalized_title"] or title["raw_title"]
            normalized = normalized.lower().strip()
            
            # Skip if we've seen this title before
            if normalized in seen_titles:
                continue
                
            seen_titles.add(normalized)
            unique_titles.append(title)
            
        return unique_titles

    def _extract_titles(self, text: str) -> List[Dict]:
        titles = []
        for pattern in self.title_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                title = match.group(1).strip()
                if title and self._is_valid_title(title):
                    matched = self._match_title(title)
                    entry = {
                        "raw_title": title,
                        "normalized_title": matched["title"] if matched else None,
                        "field": matched.get("field") if matched else None,
                        "level": self._extract_level(title),
                        "confidence": self._calculate_confidence(match)
                    }
                    titles.append(entry)
        
        # Deduplicate titles
        return self._deduplicate_titles(titles)

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

    def _calculate_confidence(self, match) -> float:
        # Calculate confidence based on match position and context
        confidence = 0.8  # Base confidence
        
        # Adjust based on match position (earlier matches are more likely to be titles)
        if match.start() < 1000:  # Within first 1000 characters
            confidence += 0.1
            
        # Adjust based on surrounding context
        context = match.group(0).lower()
        if any(kw in context for kw in ["looking for", "seeking", "hiring", "position of"]):
            confidence += 0.1
            
        return min(confidence, 1.0)

    def _extract_experience_requirements(self, text: str) -> List[Dict]:
        experience_entries = []
        seen_requirements: Set[str] = set()
        
        # Split text into sentences or bullet points
        sentences = re.split(r'[.•\n]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Try to match experience in this sentence
            match = self.experience_matcher.match(sentence)
            if match:
                # Create a unique key for deduplication
                requirement_key = f"{match.years}_{match.normalized_title or match.title}"
                if requirement_key in seen_requirements:
                    continue
                    
                seen_requirements.add(requirement_key)
                
                entry = {
                    "title": match.title,
                    "normalized_title": match.normalized_title,
                    "field": match.field,
                    "min_years": match.years,
                    "required": match.required,
                    "preferred": match.preferred,
                    "confidence": match.confidence,
                    "skills": match.skills,
                    "level": match.level
                }
                experience_entries.append(entry)

        return experience_entries

    def extract(self, job_text: str) -> Dict:
        job_text = job_text.lower()
        
        # Extract titles and experience requirements
        titles = self._extract_titles(job_text)
        experience_requirements = self._extract_experience_requirements(job_text)
        
        return {
            "job_titles": titles,
            "experience_requirements": experience_requirements
        }

def main():
    job_file_path = Path("inputs/jobs/job1.txt")
    experience_db_path = "data/experience.json"

    # Check if job file exists
    if not job_file_path.exists():
        print(f"❌ Job description file not found: {job_file_path}")
        return

    # Load job description
    job_text = job_file_path.read_text(encoding="utf-8")
    print(job_text)
    print("--------------------------------")

    # Initialize extractor
    extractor = JobExperienceExtractor(experience_file=experience_db_path)

    # Extract information
    extracted = extractor.extract(job_text)

    # Display results
    if extracted["job_titles"]:
        print(f"\n✅ Extracted {len(extracted['job_titles'])} job title(s):\n")
        for i, item in enumerate(extracted["job_titles"], start=1):
            print(f"🔹 Title {i}:\n{json.dumps(item, indent=2)}\n")
            
    if extracted["experience_requirements"]:
        print(f"\n✅ Extracted {len(extracted['experience_requirements'])} experience requirement(s):\n")
        for i, item in enumerate(extracted["experience_requirements"], start=1):
            print(f"🔹 Experience {i}:\n{json.dumps(item, indent=2)}\n")
            
    if not extracted["job_titles"] and not extracted["experience_requirements"]:
        print("⚠️ No job titles or experience requirements found.")

if __name__ == "__main__":
    main()