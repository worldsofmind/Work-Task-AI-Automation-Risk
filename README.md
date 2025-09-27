
# AI Task Risk Analyzer (spaCy + SSG)

This Streamlit app:
1) Extracts **work tasks** from job descriptions using rules + **dependency-parse verb–object** mining (spaCy).
2) Aligns them to **SkillsFuture Singapore (SSG) key tasks**.
3) Scores **routine-ness** and **AI replaceability**, with explanations and a recommended operating mode.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Push this folder to a GitHub repo.
2. Create a new app pointing to `app.py` on `main`.
3. The `requirements.txt` includes the spaCy English model via a wheel URL so it installs automatically.

## Data files
- `data/ssg_key_tasks.csv` — canonical SSG key tasks (`sector, role, task_text`).
- `data/cue_dicts.json` — verbs, cues, and AI pattern hints.
- `data/sample_job_description.txt` — a quick JD to test.

## Tuning
- Similarity threshold for SSG match: in `app/utils.py` change `accepted = top["similarity"] >= 0.72`.
- Score bump when matched to SSG: `adjust_scores_with_ssg()`.
- Add verbs/cues in `data/cue_dicts.json`.
