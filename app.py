# app.py  (flat layout: all files in repo root)
import streamlit as st
import pandas as pd

# Import from flat utils.py (no package/folder needed)
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

st.set_page_config(page_title="AI Task Risk Analyzer (spaCy + SSG)", layout="wide")

# --- Cached loaders for flat files in repo root ---
@st.cache_resource
def get_cues():
    # cue_dicts.json is in the repo root
    return load_cues("cue_dicts.json")

@st.cache_resource
def get_ssg_index():
    # ssg_key_tasks.csv is in the repo root
    df = pd.read_csv("ssg_key_tasks.csv")
    return SSGTaskIndex(df[["sector", "role", "task_text"]])

cues = get_cues()
ssg_index = get_ssg_index()

st.title("AI Task Risk Analyzer")
st.caption("Dependency-parse verb–object extraction (spaCy) + SSG key task matching — flat layout (no folders).")

with st.sidebar:
    st.subheader("Optional hints")
    sector_hint = st.text_input("Sector hint")
    role_hint = st.text_input("Role hint")

tab1, tab2 = st.tabs(["Analyze JD", "Batch CSV"])

def analyze_text(jd_text: str) -> pd.DataFrame:
    tasks = extract_tasks(jd_text, cues)
    rows = []
    for t in tasks:
        ssg_info = enrich_with_ssg(t, ssg_index, sector_hint or None, role_hint or None)
        r_score, r_why = score_routine(t, cues)
        a_score, a_detail = score_ai_replaceability(t, cues)
        r_score, a_score = adjust_scores_with_ssg(r_score, a_score, ssg_info)

        rows.append({
            "task_text": t,
            "canonical_task": ssg_info["canonical_task"] or "",
            "sector": ssg_info["sector"] or "",
            "role": ssg_info["role"] or "",
            "ssg_match_sim": ssg_info["match_sim"],
            "task_cluster": cluster_label(t),
            "routine_score": r_score,
            "ai_replaceability_score": a_score,
            "mode": recommend_mode(a_score),
            "routine_cues": ", ".join(r_why["routine_cues_matched"]) or "-",
            "predictable_cues": ", ".join(r_why["predictable_cues_matched"]) or "-",
            "dimension_breakdown": a_detail,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["ai_replaceability_score", "routine_score"], ascending=False).reset_index(drop=True)
    return df

with tab1:
    colA, colB = st.columns([1, 1])
    with colA:
        # sample_job_description.txt is in the repo root
        try:
            default_text = open("sample_job_description.txt", "r", encoding="utf-8").read()
        except Exception:
            default_text = ""
        jd_input = st.text_area("Paste a Job Description", height=260, value=default_text)
        uploaded = st.file_uploader("...or upload a .txt file", type=["txt"], key="txt")
        if uploaded is not None:
            jd_input = uploaded.read().decode("utf-8", errors="ignore")
        run = st.button("Extract & Score Tasks", type="primary")

    with colB:
        st.markdown("#### Results")
        if run and jd_input.strip():
            df = analyze_text(jd_input)
            if df.empty:
                st.info("No task-like lines found.")
            else:
                st.dataframe(
                    df[[
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
                    ]],
                    use_container_width=True,
                    hide_index=True,
                )

                for _, row in df.iterrows():
                    with st.expander(f"Why: {row['task_text']}"):
                        st.write("**AI dimensions**")
                        st.json(row["dimension_breakdown"])

                st.download_button(
                    "Download CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="ai_task_risk_results.csv",
                    mime="text/csv",
                )

with tab2:
    st.markdown("Upload a CSV with a `jd_text` column; results will include SSG matches.")
    csv_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
    if csv_file is not None:
        df_in = pd.read_csv(csv_file)
        if "jd_text" not in df_in.columns:
            st.error("CSV must contain a 'jd_text' column.")
        else:
            if st.button("Run Batch"):
                out_rows = []
                for _, r in df_in.iterrows():
                    jd = str(r["jd_text"] or "")
                    df = analyze_text(jd)
                    if df.empty:
                        continue
                    for _, tr in df.iterrows():
                        merged = {
                            **r.to_dict(),
                            **{
                                "task_text": tr["task_text"],
                                "canonical_task": tr["canonical_task"],
                                "sector": tr["sector"],
                                "role": tr["role"],
                                "ssg_match_sim": tr["ssg_match_sim"],
                                "task_cluster": tr["task_cluster"],
                                "routine_score": tr["routine_score"],
                                "ai_replaceability_score": tr["ai_replaceability_score"],
                                "mode": tr["mode"],
                            },
                        }
                        out_rows.append(merged)

                if not out_rows:
                    st.info("No task-like lines found in batch.")
                else:
                    df_out = pd.DataFrame(out_rows)
                    st.dataframe(df_out.head(50), use_container_width=True)
                    st.download_button(
                        "Download Batch Results",
                        data=df_out.to_csv(index=False).encode("utf-8"),
                        file_name="batch_ai_task_risk_results.csv",
                        mime="text/csv",
                    )
