# app.py — simple, business-friendly UI (flat layout)
import streamlit as st
import pandas as pd

from utils import (
    load_cues,
    extract_tasks,
    score_routine,
    score_ai_replaceability,
    recommend_mode,
    cluster_label,
    enrich_with_ssg,
    adjust_scores_with_ssg,
    SSGTaskIndex,
)

st.set_page_config(page_title="Job Task AI Readiness", layout="wide")

# =========================
# Overview (plain English)
# =========================
st.title("Job Task AI Readiness")
st.markdown(
    """
**What this tool does**

- Reads a job description and pulls out the **work tasks**.
- Matches each task to **SkillsFuture Singapore** “key tasks” where possible.
- Scores each task for:
  - **Routine** (how repeatable/standardised it is)
  - **AI replaceability** (how likely parts could be automated)
- Recommends a **working mode** for each task:
  - *Assist-only*, *Assist (Human-in-the-loop)*, or *Autonomous with QA*.

**Who it’s for**  
Business and HR users who want a quick, easy view of where work can be supported or automated.
"""
)

# =============
# Cached data
# =============
@st.cache_resource
def get_cues():
    return load_cues("cue_dicts.json")

@st.cache_resource
def get_ssg_index():
    df = pd.read_csv("ssg_key_tasks.csv")
    return SSGTaskIndex(df[["sector", "role", "task_text"]])

cues = get_cues()
ssg_index = get_ssg_index()

# =========================
# Sidebar: quick help
# =========================
with st.sidebar:
    st.header("Quick steps")
    st.write("1) Paste a Job Description")
    st.write("2) (Optional) Add tasks we missed")
    st.write("3) Click **Analyze**")
    st.write("4) Review scores & download CSV")
    st.divider()
    st.subheader("Optional filters")
    sector_hint = st.text_input("Sector hint (optional)")
    role_hint = st.text_input("Role hint (optional)")
    st.caption("If you know the sector or role, add it here to improve matching.")

# =========================
# Inputs
# =========================
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("1) Paste the Job Description")
    try:
        default_text = open("sample_job_description.txt", "r", encoding="utf-8").read()
    except Exception:
        default_text = ""
    jd_text = st.text_area(
        "Job description text",
        value=default_text,
        height=260,
        placeholder="Paste the JD here…",
    )
    upload = st.file_uploader("…or upload a .txt file", type=["txt"])
    if upload is not None:
        jd_text = upload.read().decode("utf-8", errors="ignore")

with col_right:
    st.subheader("2) Tasks we missed (optional)")
    st.caption("If the extractor skipped any real **work tasks**, list them **one per line**.")
    manual_tasks_text = st.text_area(
        "Add missing tasks (one per line)",
        value="",
        height=140,
        placeholder="e.g.\nConduct quarterly supplier performance reviews\nPrepare incident response playbooks",
    )
    st.caption("We’ll include these tasks in the same analysis.")

run = st.button("Analyze", type="primary")

# =========================
# Analysis logic
# =========================
def analyze_text(jd_text: str, manual_tasks_text: str) -> pd.DataFrame:
    # Extract tasks from JD
    extracted = extract_tasks(jd_text or "", cues)

    # Add manual tasks (dedupe, keep non-empty)
    manual = [t.strip() for t in (manual_tasks_text or "").split("\n")]
    manual = [t for t in manual if t]
    # de-duplicate (case-insensitive)
    seen = {t.lower() for t in extracted}
    for t in manual:
        if t.lower() not in seen:
            extracted.append(t)
            seen.add(t.lower())

    # Score + SSG match
    rows = []
    for t in extracted:
        ssg_info = enrich_with_ssg(t, ssg_index, sector_hint or None, role_hint or None)
        r_score, r_why = score_routine(t, cues)
        a_score, a_detail = score_ai_replaceability(t, cues)
        r_score, a_score = adjust_scores_with_ssg(r_score, a_score, ssg_info)
        rows.append(
            {
                "task_text": t,
                "canonical_task": ssg_info.get("canonical_task") or "",
                "sector": ssg_info.get("sector") or "",
                "role": ssg_info.get("role") or "",
                "ssg_match_sim": ssg_info.get("match_sim"),
                "task_cluster": cluster_label(t),
                "routine_score": r_score,
                "ai_replaceability_score": a_score,
                "mode": recommend_mode(a_score),
                "routine_cues": ", ".join(r_why.get("routine_cues_matched", [])) or "-",
                "predictable_cues": ", ".join(r_why.get("predictable_cues_matched", [])) or "-",
                "dimension_breakdown": a_detail,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by=["ai_replaceability_score", "routine_score"], ascending=False
        ).reset_index(drop=True)
    return df

# =========================
# Results
# =========================
if run:
    if not (jd_text or manual_tasks_text):
        st.warning("Please paste a Job Description or add at least one task.")
    else:
        df = analyze_text(jd_text, manual_tasks_text)

        st.subheader("3) Results")

        st.markdown(
            """
**How to read this:**
- **Task** — a short, action-oriented activity.
- **Canonical task** — a close match from **SkillsFuture Singapore** (if any).
- **Routine** — higher = more repeatable/standard. Think checklists, SOPs, schedules.
- **AI replaceability** — higher = more likely parts can be automated with today’s tools.
- **Mode** — suggested way to run the task:
  - **Assist-only**: AI helps with steps; a person drives.
  - **Assist (HITL)**: AI does most; a person reviews/approves.
  - **Autonomous with QA**: AI can run the task; a person spot-checks.
"""
        )

        if df.empty:
            st.info("No task-like lines found. Try adding tasks in the box on the right.")
        else:
            st.dataframe(
                df[
                    [
                        "task_text",
                        "canonical_task",
                        "sector",
                        "role",
                        "ssg_match_sim",
                        "task_cluster",
                        "routine_score",
                        "ai_replaceability_score",
                        "mode",
                        "routine_cues",
                        "predictable_cues",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("See explanation details for each task"):
                for _, row in df.iterrows():
                    st.markdown(f"**Task**: {row['task_text']}")
                    st.markdown(
                        f"- **Why these scores?**  \n"
                        f"  • Routine cues: {row['routine_cues'] if row['routine_cues'] != '-' else '—'}  \n"
                        f"  • Predictable cues: {row['predictable_cues'] if row['predictable_cues'] != '-' else '—'}  \n"
                        f"  • Detailed AI view: "
                    )
                    st.json(row["dimension_breakdown"])
                    st.divider()

            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="ai_task_risk_results.csv",
                mime="text/csv",
            )

# =========================
# Footer tips
# =========================
st.divider()
st.markdown(
    """
**Tips**
- If the table looks empty or off, add a few missing tasks in the **Tasks we missed** box and click **Analyze** again.
- Adding a **Sector hint** or **Role hint** in the sidebar can improve the canonical task matches.
"""
)

