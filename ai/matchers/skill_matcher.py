from typing import List, Dict
from rapidfuzz import fuzz

class SkillMatcher:
    def __init__(self, threshold: int = 85):
        self.threshold = threshold

    def match(self, resume_skills: List[Dict], job_skills: List[Dict]) -> Dict:
        matched_skills = []
        resume_set = {s["normalized_name"]: s for s in resume_skills}
        job_set = {s["normalized_name"]: s for s in job_skills}

        matched_count = 0

        for job_skill_name, job_skill in job_set.items():
            best_match = None
            best_score = 0
            reason = "not found"

            for resume_skill_name, resume_skill in resume_set.items():
                score = fuzz.token_sort_ratio(job_skill_name, resume_skill_name)
                if score > best_score:
                    best_score = score
                    best_match = resume_skill

            if best_score >= self.threshold:
                matched_count += 1
                reason = "fuzzy match" if best_score < 100 else "exact match"
                matched_skills.append({
                    "job_skill": job_skill["original_name"],
                    "resume_skill": best_match["original_name"],
                    "category": job_skill["category"],
                    "score": best_score,
                    "reason": reason
                })
            else:
                matched_skills.append({
                    "job_skill": job_skill["original_name"],
                    "resume_skill": None,
                    "category": job_skill["category"],
                    "score": 0,
                    "reason": "missing"
                })

        overall_score = round((matched_count / len(job_skills)) * 100, 2) if job_skills else 0

        return {
            "match_percentage": overall_score,
            "matched_skills": matched_skills
        }
        
if __name__ == "__main__":
    from ai.extractors.resume.skill_extractor import ResumeSkillExtractor
    from ai.extractors.job.skill_extractor import JobSkillExtractor
    from pathlib import Path

    # Load resume and job text
    resume_text = Path("inputs/resumes/resume1.txt").read_text(encoding="utf-8")
    job_text = Path("inputs/jobs/job1.txt").read_text(encoding="utf-8")
    
    print(resume_text)
    print("--------------------------------")
    print(job_text)

    # Extract skills
    resume_skills = ResumeSkillExtractor("data/skills.json").extract(resume_text)
    job_skills = JobSkillExtractor("data/skills.json").extract(job_text)

    # Match skills
    matcher = SkillMatcher()
    result = matcher.match(resume_skills, job_skills)

    print(f"\nOverall Match: {result['match_percentage']}%\n")
    for match in result["matched_skills"]:
        print(f"- {match['job_skill']} → {match['resume_skill'] or '❌ Not Found'} "
                f"(Score: {match['score']}, Reason: {match['reason']})")

