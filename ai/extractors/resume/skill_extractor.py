import json
import re
from pathlib import Path
from typing import List, Dict
from rapidfuzz import fuzz, process
import spacy

class ResumeSkillExtractor:
    def __init__(self, skill_file_path: str = "data/skills.json", fuzzy_threshold: int = 85):
        self.skills = self._load_skills(skill_file_path)
        self.skill_lookup = self._build_lookup()
        self.fuzzy_threshold = fuzzy_threshold
        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])

    def _load_skills(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_lookup(self) -> Dict[str, Dict]:
        lookup = {}
        for skill in self.skills:
            keys = set([skill["name"], skill["normalized_name"]] + skill.get("aliases", []))
            for k in keys:
                lookup[k.strip().lower()] = skill
        return lookup

    def _normalize(self, text: str) -> str:
        return re.sub(r"[^\w\s\-\.]", "", text).strip().lower()

    def extract(self, resume_text: str) -> List[Dict]:
        doc = self.nlp(resume_text)
        candidates = set()

        for token in doc:
            if not token.is_stop and not token.is_punct and len(token.text) > 2:
                candidates.add(self._normalize(token.text))
        for chunk in doc.noun_chunks:
            candidates.add(self._normalize(chunk.text))

        matches = []
        seen_normalized = set()  

        for candidate in candidates:
            best, score, _ = process.extractOne(candidate, self.skill_lookup.keys(), scorer=fuzz.token_sort_ratio)
            if score >= self.fuzzy_threshold:
                skill = self.skill_lookup[best]
                norm = skill["normalized_name"]

                if norm not in seen_normalized:  
                    seen_normalized.add(norm)
                    matches.append({
                        "id": skill["id"],
                        "matched_text": candidate,
                        "normalized_name": norm,
                        "original_name": skill["name"],
                        "category": skill["category"],
                        "subcategory": skill["subcategory"],
                        "aliases": skill.get("aliases", []),
                        "tags": skill.get("tags", []),
                        "related_skills": skill.get("related_skills", []),
                        "score": score,
                    })

        return sorted(matches, key=lambda x: x["score"], reverse=True)



def main():
    from pathlib import Path
    resume_path = Path("inputs/resumes/resume1.txt")  # Or any resume file
    if not resume_path.exists():
        print("Resume file not found.")
        return

    resume_text = resume_path.read_text(encoding="utf-8")
    extractor = ResumeSkillExtractor(skill_file_path="data/skills.json")
    results = extractor.extract(resume_text)

    print(f"\nExtracted {len(results)} skill(s):\n")
    for skill in results:
        print(f"- {skill['matched_text']} → {skill['original_name']} (score: {skill['score']})")
        print(f"  Category: {skill['category']} | Subcategory: {skill['subcategory']}")
        print(f"  Tags: {', '.join(skill['tags'])}")
        print()

if __name__ == "__main__":
    main()
