import json
import re
from typing import List, Dict, Optional
from pathlib import Path


class JobEducationExtractor:
    def __init__(self, education_file: str = "data/education.json"):
        self.education_data = self._load_education_data(education_file)
        self.degree_lookup = self._build_degree_lookup()

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

    def extract(self, job_text: str) -> List[Dict]:
        text = job_text.lower()
        results = []

        for alias, degree in self.degree_lookup.items():
            if alias in text:
                required = "required" in self._surrounding_text(text, alias, window=50)
                preferred = "preferred" in self._surrounding_text(text, alias, window=50)
                major = self._extract_major(text, alias)

                entry = {
                    "degree": degree,
                    "major": major,
                    "required": required,
                    "preferred": preferred,
                    "degree_level": self._infer_level(degree),
                }

                if not any(e["degree"] == degree for e in results):
                    results.append(entry)

        return results

    def _surrounding_text(self, text: str, keyword: str, window: int = 50) -> str:
        idx = text.find(keyword)
        start = max(0, idx - window)
        end = idx + len(keyword) + window
        return text[start:end]

    def _infer_level(self, degree: str) -> str:
        degree = degree.lower()
        if "phd" in degree or "doctor" in degree:
            return "phd"
        elif "master" in degree or "mba" in degree:
            return "master"
        elif "bachelor" in degree:
            return "bachelor"
        elif "associate" in degree:
            return "associate"
        return "unspecified"

    def _extract_major(self, text: str, alias: str) -> Optional[str]:
        patterns = [
            fr"{alias} (in|of)\s+([\w\s&]+)",
            fr"(degree|background) in\s+([\w\s&]+)"
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(2).strip().title()
        return None
    
def main():
    job_path = Path("inputs/jobs/job1.txt")
    if not job_path.exists():
        print("❌ Job description not found:", job_path)
        return

    job_text = job_path.read_text(encoding="utf-8")

    extractor = JobEducationExtractor("data/education.json")
    education_requirements = extractor.extract(job_text)

    if education_requirements:
        print(f"\n Extracted {len(education_requirements)} education requirement(s):\n")
        for entry in education_requirements:
            print(json.dumps(entry, indent=2))
    else:
        print(" No education requirements found in the job description.")

if __name__ == "__main__":
    main()
