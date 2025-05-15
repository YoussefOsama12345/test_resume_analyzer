from typing import List, Dict, Optional
from rapidfuzz import fuzz
from ai.extractors.resume.education_extractor import ResumeEducationExtractor
from ai.extractors.job.education_extractor import JobEducationExtractor
from pathlib import Path
import json

class EducationMatcher:
    def __init__(self, degree_threshold: int = 90, major_threshold: int = 80):
        self.degree_threshold = degree_threshold
        self.major_threshold = major_threshold

    def match(self, resume_edu: List[Dict], job_edu: List[Dict]) -> Dict:
        results = []
        matched_count = 0

        for job_item in job_edu:
            best_match = None
            best_score = 0
            reason = "no match"

            for res_item in resume_edu:
                degree_score = fuzz.token_sort_ratio(job_item['degree'].lower(), res_item['degree'].lower())
                major_score = 0

                if job_item.get("major") and res_item.get("major"):
                    major_score = fuzz.token_sort_ratio(job_item['major'].lower(), res_item['major'].lower())

                if degree_score >= self.degree_threshold:
                    match_score = degree_score
                    if major_score > 0:
                        match_score = (degree_score + major_score) // 2

                    if match_score > best_score:
                        best_score = match_score
                        best_match = {
                            "resume_degree": res_item["degree"],
                            "resume_major": res_item.get("major"),
                            "job_degree": job_item["degree"],
                            "job_major": job_item.get("major"),
                            "required": job_item["required"],
                            "preferred": job_item["preferred"],
                            "degree_level": job_item["degree_level"],
                            "score": match_score,
                            "reason": "degree and major match" if major_score else "degree match only"
                        }

            if best_score >= self.degree_threshold:
                results.append(best_match)
                if job_item["required"]:
                    matched_count += 1
            else:
                results.append({
                    "resume_degree": None,
                    "resume_major": None,
                    "job_degree": job_item["degree"],
                    "job_major": job_item.get("major"),
                    "required": job_item["required"],
                    "preferred": job_item["preferred"],
                    "degree_level": job_item["degree_level"],
                    "score": 0,
                    "reason": "no match"
                })

        total_required = sum(1 for j in job_edu if j["required"])
        match_percentage = round((matched_count / total_required) * 100, 2) if total_required else 100.0

        return {
            "education_match_percentage": match_percentage,
            "education_matches": results
        }

if __name__ == "__main__":

    # Load text files
    resume_text = Path("inputs/resumes/resume1.txt").read_text(encoding="utf-8")
    job_text = Path("inputs/jobs/job1.txt").read_text(encoding="utf-8")

    # Extract education data
    resume_edu = ResumeEducationExtractor("data/education.json").extract(resume_text)
    job_edu = JobEducationExtractor("data/education.json").extract(job_text)
    
    print(f"\nExtracted {resume_edu} education requirement(s) from resume.")
    print(f"\nExtracted {job_edu} education requirement(s) from job description.")

    # Match education
    matcher = EducationMatcher()
    result = matcher.match(resume_edu, job_edu)

    print(f"\n Education Match: {result['education_match_percentage']}%\n")
    for match in result["education_matches"]:
        print(json.dumps(match, indent=2))
