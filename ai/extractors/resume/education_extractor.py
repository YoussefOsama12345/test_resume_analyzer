import json
import re
from typing import List, Dict, Optional
from pathlib import Path
import spacy

class ResumeEducationExtractor:
    def __init__(self, education_file: str = "data/education.json"):
        self.education_data = self._load_education_data(education_file)
        self.degree_lookup = self._build_degree_lookup()
        self.nlp = spacy.load("en_core_web_sm")

    def _load_education_data(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_degree_lookup(self) -> Dict[str, str]:
        lookup = {}
        for item in self.education_data:
            normalized = item["degree"].strip().lower()
            aliases = item.get("aliases", [])
            for alias in [normalized] + aliases:
                lookup[alias.strip().lower()] = item["degree"]
        return lookup

    def _extract_education_section(self, text: str) -> List[str]:
        lines = text.splitlines()
        edu_lines = []
        capture = False
        for line in lines:
            line_lower = line.lower()
            if "education" in line_lower:
                capture = True
                continue
            if capture:
                if any(kw in line_lower for kw in ["experience", "certification", "project", "skills", "summary"]):
                    break
                if line.strip():
                    edu_lines.append(line.strip())
        return edu_lines

    def extract(self, resume_text: str) -> List[Dict]:
        edu_lines = self._extract_education_section(resume_text)
        entries = []
        buffer = []

        # Split into blocks per degree entry
        for line in edu_lines:
            if any(deg in line.lower() for deg in self.degree_lookup):
                if buffer:
                    entries.append("\n".join(buffer))
                    buffer = []
            buffer.append(line)
        if buffer:
            entries.append("\n".join(buffer))

        # Now parse each block
        results = []
        for block in entries:
            degree = self._extract_degree(block)
            institution = self._extract_institution(block)
            major = self._extract_major(block)
            years = self._extract_years(block)
            gpa = self._extract_gpa(block)
            honors = self._extract_honors(block)

            if degree and institution:
                results.append({
                    "degree": degree,
                    "major": major,
                    "institution": institution,
                    "location": None,
                    "start_year": years[0] if years else None,
                    "end_year": years[1] if years else None,
                    "gpa": gpa,
                    "honors": honors
                })

        return results

    def _extract_degree(self, text: str) -> Optional[str]:
        text = text.lower()
        for alias, degree_name in self.degree_lookup.items():
            if alias in text:
                return degree_name
        return None

    def _extract_major(self, text: str) -> Optional[str]:
        match = re.search(r"(major(ed)? in|degree in)\s+([\w\s&]+)", text.lower())
        if match:
            return match.group(3).strip().title()
        return None

    def _extract_institution(self, text: str) -> Optional[str]:
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["ORG", "GPE"] and any(w in ent.text.lower() for w in ["university", "college", "institute"]):
                return ent.text.strip()
        return None

    def _extract_years(self, text: str) -> Optional[List[int]]:
        years = re.findall(r"(19|20)\d{2}", text)
        years = sorted(set(int(y) for y in years))
        if len(years) >= 2:
            return years[:2]
        elif len(years) == 1:
            return [years[0], None]
        return None

    def _extract_gpa(self, text: str) -> Optional[float]:
        match = re.search(r"gpa[:\s]*([0-4]\.\d{1,2})", text.lower())
        return float(match.group(1)) if match else None

    def _extract_honors(self, text: str) -> Optional[str]:
        honors_keywords = ["summa cum laude", "magna cum laude", "cum laude"]
        for keyword in honors_keywords:
            if keyword in text.lower():
                return keyword.title()
        return None

    
def main():
    # Path to the resume file
    resume_path = Path("inputs/resumes/resume1.txt")
    if not resume_path.exists():
        print(f"Resume file not found at {resume_path}")
        return

    # Read resume text
    resume_text = resume_path.read_text(encoding="utf-8")
    print(resume_text)
    print("--------------------------------")

    # Initialize extractor
    extractor = ResumeEducationExtractor(education_file="data/education.json")

    # Extract education data
    education_entries = extractor.extract(resume_text)

    # Display results
    if education_entries:
        print(f"\nExtracted {len(education_entries)} education entry(ies):\n")
        for entry in education_entries:
            print(json.dumps(entry, indent=2))
    else:
        print("No education entries found.")

if __name__ == "__main__":
    main()
    
    
