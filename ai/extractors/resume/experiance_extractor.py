import re
import json
import spacy
from typing import List, Dict, Optional, Set, Tuple
from rapidfuzz import process, fuzz
from pathlib import Path
from datetime import datetime

class ResumeExperienceExtractor:
    def __init__(self, experience_file: str = "data/experience.json"):
        self.nlp = spacy.load("en_core_web_sm")
        self.experience_db = self._load_experience_data(experience_file)
        self.title_lookup = self._build_title_lookup()
        
        # Enhanced patterns for better extraction
        self.date_patterns = [
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(?:19|20)\d{2}",
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:19|20)\d{2}",
            r"\d{1,2}/\d{1,2}/(?:19|20)\d{2}",
            r"\d{1,2}-\d{1,2}-(?:19|20)\d{2}"
        ]
        
        self.skill_patterns = [
            r"(?:proficient|expert|skilled|experienced)\s+(?:in|with|at)\s+([a-zA-Z\s,&/\\-]+)",
            r"(?:strong|solid|excellent|good|deep)\s+(?:knowledge|understanding|experience|proficiency)\s+(?:in|with|of)\s+([a-zA-Z\s,&/\\-]+)",
            r"(?:familiarity|familiar)\s+(?:with|in)\s+([a-zA-Z\s,&/\\-]+)",
            r"(?:developed|created|implemented|built|designed)\s+(?:using|with)\s+([a-zA-Z\s,&/\\-]+)",
            r"(?:utilized|used|worked with)\s+([a-zA-Z\s,&/\\-]+)"
        ]
        
        self.achievement_patterns = [
            r"(?:increased|improved|enhanced|optimized|reduced|decreased)\s+[^.]*?by\s+\d+%",
            r"(?:led|managed|supervised|coordinated)\s+[^.]*?team",
            r"(?:developed|created|implemented|built|designed)\s+[^.]*?solution",
            r"(?:achieved|attained|reached)\s+[^.]*?goal",
            r"(?:saved|reduced|decreased)\s+[^.]*?cost"
        ]

    def _load_experience_data(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_title_lookup(self) -> Dict[str, Dict]:
        lookup = {}
        for item in self.experience_db:
            key = item["title"].strip().lower()
            lookup[key] = item
            # Add common variations
            if "senior" in key:
                lookup[key.replace("senior", "sr")] = item
            if "junior" in key:
                lookup[key.replace("junior", "jr")] = item
            if "lead" in key:
                lookup[key.replace("lead", "principal")] = item
        return lookup

    def _extract_skills(self, text: str) -> List[str]:
        skills = set()
        doc = self.nlp(text)
        
        # Extract skills using patterns
        for pattern in self.skill_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                skill = match.group(1).strip().lower()
                if skill and len(skill.split()) <= 5:  # Avoid long phrases
                    skills.add(skill)
        
        # Extract technical terms using spaCy
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2:
                # Check if it's likely a technical term
                if any(char.isupper() for char in token.text) or token.text.lower() in self._get_common_tech_terms():
                    skills.add(token.text.lower())
        
        return list(skills)

    def _get_common_tech_terms(self) -> Set[str]:
        return {
            "python", "java", "javascript", "typescript", "react", "angular", "vue",
            "node", "express", "django", "flask", "spring", "aws", "azure", "gcp",
            "docker", "kubernetes", "sql", "nosql", "mongodb", "postgresql", "mysql",
            "redis", "kafka", "rabbitmq", "git", "ci", "cd", "jenkins", "terraform",
            "ansible", "linux", "unix", "agile", "scrum", "jira", "confluence"
        }

    def _extract_achievements(self, text: str) -> List[str]:
        achievements = []
        for pattern in self.achievement_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                achievement = match.group(0).strip()
                if achievement and len(achievement.split()) <= 20:  # Avoid too long achievements
                    achievements.append(achievement)
        return achievements

    def _extract_dates(self, text: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        dates = []
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    date_str = match.group(0)
                    # Try different date formats
                    for fmt in ["%B %Y", "%b %Y", "%m/%d/%Y", "%m-%d-%Y"]:
                        try:
                            date = datetime.strptime(date_str, fmt)
                            dates.append(date)
                            break
                        except ValueError:
                            continue
                except:
                    continue
        
        if not dates:
            return None, None
            
        dates.sort()
        return dates[0], dates[-1]

    def extract(self, resume_text: str) -> List[Dict]:
        experience_section = self._extract_experience_section(resume_text)
        entries = self._split_into_experience_blocks(experience_section)

        results = []
        for block in entries:
            raw_title = self._extract_title(block)
            company = self._extract_company(block)
            start_date, end_date = self._extract_dates(block)
            description = self._extract_description(block)
            skills = self._extract_skills(block)
            achievements = self._extract_achievements(block)

            if raw_title and company:
                norm_data = self._match_title(raw_title)
                entry = {
                    "job_title": raw_title,
                    "normalized_title": norm_data["title"] if norm_data else raw_title,
                    "company": company,
                    "start_date": start_date.strftime("%Y-%m") if start_date else None,
                    "end_date": end_date.strftime("%Y-%m") if end_date else None,
                    "duration_years": self._calculate_duration(start_date, end_date),
                    "description": description,
                    "skills": skills,
                    "achievements": achievements,
                    "industry": norm_data.get("industry") if norm_data else None,
                    "level": self._extract_level(raw_title),
                    "sample_responsibilities": norm_data.get("responsibilities")[:2] if norm_data else []
                }
                results.append(entry)
        return results

    def _calculate_duration(self, start_date: Optional[datetime], end_date: Optional[datetime]) -> float:
        if not start_date:
            return 0.0
        end = end_date or datetime.now()
        duration = end - start_date
        return round(duration.days / 365.25, 1)  # Convert to years with 1 decimal place

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

    def _match_title(self, title: str) -> Optional[Dict]:
        title_clean = title.lower().strip()
        
        # Try exact match first
        if title_clean in self.title_lookup:
            return self.title_lookup[title_clean]
            
        # Try fuzzy matching with different strategies
        strategies = [
            (fuzz.token_sort_ratio, 85),
            (fuzz.token_set_ratio, 90),
            (fuzz.partial_ratio, 95)
        ]
        
        for scorer, threshold in strategies:
            best_match, score, _ = process.extractOne(
                title_clean,
                self.title_lookup.keys(),
                scorer=scorer
            )
            if score >= threshold:
                return self.title_lookup[best_match]
                
        return None

    def _extract_experience_section(self, text: str) -> str:
        lines = text.splitlines()
        exp_lines = []
        capture = False
        
        section_headers = [
            "experience",
            "work experience",
            "professional experience",
            "employment history",
            "work history"
        ]
        
        section_enders = [
            "education",
            "certifications",
            "projects",
            "skills",
            "summary",
            "objective",
            "references"
        ]
        
        for line in lines:
            lower = line.lower()
            if any(header in lower for header in section_headers) and len(lower) < 40:
                capture = True
                continue
            if capture:
                if any(ender in lower for ender in section_enders):
                    break
                exp_lines.append(line.strip())
        return "\n".join(exp_lines)

    def _split_into_experience_blocks(self, text: str) -> List[str]:
        lines = text.splitlines()
        blocks = []
        current_block = []

        for line in lines:
            # Start of a new experience entry if the line looks like a job title
            if re.match(r"(?i)^([A-Z][A-Za-z &]+)$", line.strip()) or \
               re.match(r"(?i)^([A-Z][A-Za-z &]+(?:Engineer|Developer|Scientist|Analyst|Architect|Manager|Consultant|Specialist))$", line.strip()):
                if current_block:
                    blocks.append("\n".join(current_block).strip())
                    current_block = []
            current_block.append(line.strip())

        if current_block:
            blocks.append("\n".join(current_block).strip())

        return blocks

    def _extract_title(self, block: str) -> Optional[str]:
        lines = block.splitlines()
        return lines[0].strip() if len(lines) > 0 else None

    def _extract_company(self, block: str) -> Optional[str]:
        lines = block.splitlines()
        if len(lines) > 1:
            company_line = lines[1].strip()
            # Handle various company line formats
            company = company_line.split(",")[0].strip()
            return company
        return None

    def _extract_description(self, text: str) -> Optional[str]:
        lines = text.splitlines()
        bullets = []
        for line in lines:
            # Skip lines that look like dates or job titles
            if not any(re.match(pattern, line, re.IGNORECASE) for pattern in self.date_patterns) and \
               not re.match(r"(?i)^([A-Z][A-Za-z &]+)$", line.strip()):
                cleaned = line.strip("-• ")
                if cleaned:
                    bullets.append(cleaned)
        return "\n".join(bullets).strip() if bullets else None

def main():
    # Set your input file paths
    resume_path = Path("inputs/resumes/resume1.txt")
    experience_db_path = "data/experience.json"

    # Check if resume file exists
    if not resume_path.exists():
        print(f"❌ Resume file not found at {resume_path}")
        return

    # Load resume text
    resume_text = resume_path.read_text(encoding="utf-8")

    # Initialize the extractor with experience.json
    extractor = ResumeExperienceExtractor(experience_file=experience_db_path)

    # Extract experiences
    experiences = extractor.extract(resume_text)

    # Display results
    if experiences:
        print(f"\n✅ Extracted {len(experiences)} experience entry(ies):\n")
        for i, entry in enumerate(experiences, start=1):
            print(f"🔹 Entry {i}:\n{json.dumps(entry, indent=2)}\n")
    else:
        print("⚠️ No experience entries found.")

if __name__ == "__main__":
    main()
